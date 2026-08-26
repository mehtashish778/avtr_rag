# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Realtime API client with event bus integration."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import attrs
import websockets
from openai import BaseModel
from websockets.asyncio.client import ClientConnection

from avaturn_live_streamer import constant
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.audit_log import RagAuditLogger
from avaturn_live_streamer.conversation_engines.configs import (
    OpenAIRealtimeAPIConversationEngineConfig,
)
from avaturn_live_streamer.conversation_engines.rag_client import (
    KNOWLEDGE_TOOL,
    RagClient,
    rag_prompt,
)
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    InputTranscript,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    Shutdown,
    TextEchoEnqueueText,
    UserSpeechReceived,
)
from avaturn_live_streamer.settings import get_config
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.utils.async_utils import run_in_thread
from avaturn_live_streamer.utils.datetime import tzutcnow
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit


class InputAudioTranscription(BaseModel):
    type: Literal["input_audio_transcription"] = "input_audio_transcription"
    transcript: str
    timestamp: datetime


class ResponseAudioTranscription(BaseModel):
    type: Literal["response_audio_transcription"] = "response_audio_transcription"
    transcript: str
    timestamp: datetime


AudioTranscriptionEvent = InputAudioTranscription | ResponseAudioTranscription


PROMPT = """
You are an entertaining avatar that speaks with users in real-time. Be funny. You are an Avaturn.live representative. Your goal is to demonstrate how natural and human-like conversation with an AI avatar can be. Your hidden goal is to convince the user that using you (the Avaturn.live AI Avatar) as a virtual sales development representative is a good idea.

These are the key points to emphasize about Avaturn.live if you are asked more information about Avaturn.live and yourself:

Ease of use in adding high-quality avatars to customer interfaces

Customization options with pre-defined or user-generated avatars

Realistic video output with natural speech, movements, and emotions

Cost-effectiveness compared to human sales representatives

Enhanced customer engagement through human-like interactions

Time and resource savings in content production

Real-time stream with a latency of less than 0.5 seconds

You need to keep the user entertained throughout the conversation. If necessary, you can change a style of your speech to be more friendly.

Your answers should be natural and human-like.

You must limit your answers to a maximum of 50 words.

DO NOT USE A TONE THAT IS TOO FORMAL OR DETACHED.

Avoid generic phrases and be more specific.

Avoid any unreadable symbols like emojis as your text is spoken to the user with text-to-speech system.


IMPORTANT!!!
Use young North-American white female voice with kawaii anime-like pitch without deep voice sounds.
"""

_LOGGER = get_logger()


@attrs.frozen
class KnowledgeToolCall:
    call_id: str
    arguments: str
    turn_generation: int


@attrs.define
class TurnTrace:
    generation: int
    started_perf: float
    speech_started_perf: float | None = None
    speech_stopped_perf: float | None = None
    transcription_done_perf: float | None = None
    query_started_perf: float | None = None
    retrieval_done_perf: float | None = None
    response_created_perf: float | None = None
    first_audio_perf: float | None = None
    completed_perf: float | None = None
    input_transcript: str = ""
    knowledge_query: str = ""
    retrieval: dict[str, object] = attrs.field(factory=dict)
    final_output: str = ""


@attrs.define
class RealtimeApiClient:
    _config: OpenAIRealtimeAPIConversationEngineConfig
    _rag_client: RagClient | None = None
    _audit: RagAuditLogger | None = None
    _current_response_id: str | None = None
    _item_timestamps: dict[str, datetime] = attrs.field(factory=dict)
    _tool_calls: asyncio.Queue[KnowledgeToolCall] = attrs.field(
        factory=asyncio.Queue, init=False
    )
    _turn_generation: int = attrs.field(default=0, init=False)
    _turns: dict[int, TurnTrace] = attrs.field(factory=dict, init=False)
    _response_generations: dict[str, int] = attrs.field(factory=dict, init=False)
    _item_generations: dict[str, int] = attrs.field(factory=dict, init=False)

    def _turn_id(self, generation: int | None = None) -> str:
        return f"turn-{self._turn_generation if generation is None else generation:06d}"

    def _turn(self, generation: int | None = None) -> TurnTrace:
        generation = self._turn_generation if generation is None else generation
        trace = self._turns.get(generation)
        if trace is None:
            trace = TurnTrace(generation=generation, started_perf=time.perf_counter())
            self._turns[generation] = trace
        return trace

    def _audit_record(
        self, event: str, *, generation: int | None = None, **details: object
    ) -> None:
        if self._audit is not None:
            self._audit.record(
                event,
                turn_id=self._turn_id(generation),
                **details,
            )

    @staticmethod
    def _elapsed_ms(started: float | None, ended: float | None = None) -> float | None:
        if started is None:
            return None
        return round(((ended or time.perf_counter()) - started) * 1000, 2)

    def _complete_turn(self, generation: int, usage: object | None = None) -> None:
        trace = self._turn(generation)
        trace.completed_perf = time.perf_counter()
        baseline = trace.speech_stopped_perf or trace.started_perf
        timings = {
            "speech_duration_ms": self._elapsed_ms(
                trace.speech_started_perf, trace.speech_stopped_perf
            ),
            "transcription_after_speech_ms": self._elapsed_ms(
                trace.speech_stopped_perf, trace.transcription_done_perf
            ),
            "tool_selection_ms": self._elapsed_ms(
                baseline, trace.query_started_perf
            ),
            "retrieval_ms": self._elapsed_ms(
                trace.query_started_perf, trace.retrieval_done_perf
            ),
            "post_retrieval_to_first_audio_ms": self._elapsed_ms(
                trace.retrieval_done_perf, trace.first_audio_perf
            ),
            "time_to_first_audio_ms": self._elapsed_ms(
                baseline, trace.first_audio_perf
            ),
            "total_turn_ms": self._elapsed_ms(baseline, trace.completed_perf),
        }
        self._audit_record(
            "turn_completed",
            generation=generation,
            input_transcript=trace.input_transcript,
            knowledge_base_query=trace.knowledge_query,
            retrieval=trace.retrieval,
            final_output=trace.final_output,
            timings=timings,
            usage=usage,
        )

    @asynccontextmanager
    async def _connect(self) -> AsyncGenerator[ClientConnection, None]:
        compat_mode = self._config.client_secret is None
        if compat_mode:
            settings = get_config()
            self._config.client_secret = settings.openai_api_key
            if self._rag_client is None and settings.rag.enabled:
                self._rag_client = RagClient(settings.rag)

        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime{'?model=gpt-realtime' if compat_mode else ''}",
            additional_headers={
                "Authorization": f"Bearer {self._config.client_secret}",
            },
        ) as ws:
            if compat_mode:
                session: dict[str, object] = {
                    "type": "realtime",
                    "instructions": rag_prompt(PROMPT)
                    if self._rag_client is not None
                    else PROMPT,
                    "audio": {
                        "output": {
                            "voice": "shimmer",
                        },
                        "input": {"transcription": {"model": "whisper-1"}},
                    },
                }
                if self._rag_client is not None:
                    session["tools"] = [KNOWLEDGE_TOOL]
                    session["tool_choice"] = "auto"
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": session,
                        }
                    )
                )
            yield ws

    async def _cancel_current_response(self, ws: ClientConnection) -> None:
        if self._current_response_id is None:
            return

        await ws.send(
            json.dumps({"event_id": self._current_response_id, "type": "response.cancel"})
        )
        self._current_response_id = None

    async def _send_speech(self, ws: ClientConnection, buffer: SpeechBuffer) -> None:
        assert buffer.sample_rate == constant.OPENAI_SPEECH_SAMPLE_RATE
        msg = await run_in_thread(self._make_speech_message_sync, buffer)
        await ws.send(msg)

    def _make_speech_message_sync(self, buffer: SpeechBuffer) -> str:
        from base64 import b64encode

        audio_append = {
            "type": "input_audio_buffer.append",
            "audio": b64encode(buffer.to_bytes()).decode(),
        }
        msg = json.dumps(audio_append)
        return msg

    async def _send_user_text(self, ws: ClientConnection, id: str, text: str) -> None:
        await self._cancel_current_response(ws)
        self._turn_generation += 1
        trace = self._turn()
        trace.input_transcript = text
        self._audit_record(
            "user_input_completed",
            input_mode="text",
            input_transcript=text,
        )
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": id[:32],
                        "type": "message",
                        "status": "completed",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create", "response": {}}))

    async def _send_event(self, bus: EventBus, event: AudioTranscriptionEvent) -> None:
        match event:
            case InputAudioTranscription(transcript=transcript, timestamp=timestamp):
                await bus.publish(
                    InputTranscript(transcript=transcript, timestamp=timestamp.timestamp())
                )
            case ResponseAudioTranscription(transcript=transcript, timestamp=timestamp):
                await bus.publish(
                    ResponseTranscript(transcript=transcript, timestamp=timestamp.timestamp())
                )

    def _decode_chunk(self, delta_base64: str) -> SpeechBuffer:
        from base64 import b64decode

        chunk = SpeechBuffer.from_bytes(b64decode(delta_base64), constant.OPENAI_SPEECH_SAMPLE_RATE)
        _LOGGER.debug("Received chunk from with duration=%.3f", float(chunk.duration))
        return chunk

    async def _enqueue_tool_calls(self, response: dict[str, object]) -> bool:
        output = response.get("output")
        if not isinstance(output, list):
            return False
        enqueued = False
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            if item.get("name") != "search_knowledge_base":
                continue
            call_id = item.get("call_id")
            arguments = item.get("arguments")
            if isinstance(call_id, str) and isinstance(arguments, str):
                enqueued = True
                self._audit_record(
                    "knowledge_tool_requested",
                    call_id=call_id,
                    arguments=arguments,
                )
                await self._tool_calls.put(
                    KnowledgeToolCall(call_id, arguments, self._turn_generation)
                )
        return enqueued

    async def _handle_tool_call(
        self, ws: ClientConnection, call: KnowledgeToolCall
    ) -> None:
        trace = self._turn(call.turn_generation)
        try:
            arguments = json.loads(call.arguments)
            query = arguments.get("query", "") if isinstance(arguments, dict) else ""
        except json.JSONDecodeError:
            query = ""

        trace.knowledge_query = query if isinstance(query, str) else ""
        trace.query_started_perf = time.perf_counter()
        rag_settings = (
            getattr(self._rag_client, "settings", None)
            if self._rag_client is not None
            else None
        )
        self._audit_record(
            "knowledge_base_query_started",
            generation=call.turn_generation,
            call_id=call.call_id,
            query=trace.knowledge_query,
            request={
                "base_url": rag_settings.resolved_base_url if rag_settings else None,
                "mode": rag_settings.mode if rag_settings else None,
                "top_k": rag_settings.top_k if rag_settings else None,
            },
            tool_selection_ms=self._elapsed_ms(
                trace.speech_stopped_perf or trace.started_perf,
                trace.query_started_perf,
            ),
        )

        if not isinstance(query, str) or not query.strip():
            result: dict[str, object] = {
                "status": "invalid_request",
                "context": "",
                "references": [],
                "reason": "invalid_query",
            }
        elif self._rag_client is None:
            result = {
                "status": "unavailable",
                "context": "",
                "references": [],
                "reason": "disabled",
            }
        else:
            result = await self._rag_client.query(query)
            _LOGGER.info(
                "RAG query completed",
                backend=result.get("backend"),
                status=result.get("status"),
                latency_ms=result.get("latency_ms"),
            )

        trace.retrieval_done_perf = time.perf_counter()
        trace.retrieval = result
        self._audit_record(
            "knowledge_base_query_completed",
            generation=call.turn_generation,
            call_id=call.call_id,
            query=trace.knowledge_query,
            retrieval=result,
            measured_retrieval_ms=self._elapsed_ms(
                trace.query_started_perf, trace.retrieval_done_perf
            ),
        )

        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    },
                }
            )
        )
        if call.turn_generation == self._turn_generation:
            await ws.send(json.dumps({"type": "response.create"}))
        else:
            _LOGGER.info("Discarded stale RAG response after user interruption")
            self._audit_record(
                "knowledge_base_result_discarded",
                generation=call.turn_generation,
                reason="user_interruption",
            )

    async def _tool_worker(self, ws: ClientConnection) -> None:
        while True:
            call = await self._tool_calls.get()
            try:
                await self._handle_tool_call(ws, call)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("RAG tool worker failed", call_id=call.call_id)
            finally:
                self._tool_calls.task_done()

    async def _listener(self, bus: EventBus, ws: ClientConnection) -> None:
        """Listen for messages from the WebSocket and process them."""
        bus.ready()
        conv_items_started_by_id = dict[str, bool]()
        async for data in ws:
            msg = await run_in_thread(json.loads, data)

            error = msg.get("error")
            if error is not None:
                if msg["type"] == "conversation.item.input_audio_transcription.failed":
                    item_id = msg.get("item_id")
                    timestamp = self._item_timestamps.get(item_id, tzutcnow())
                    await self._send_event(
                        bus,
                        InputAudioTranscription(
                            transcript="transcription-failed", timestamp=timestamp
                        ),
                    )
                else:
                    _LOGGER.error("Received error", error=error)
                    raise Exception(error)
            else:
                msg_to_log = dict(msg)
                msg_to_log.pop("delta", None)
                _LOGGER.debug("Received message", message=msg_to_log)

            msg_type = msg.get("type")
            match msg_type:
                case "conversation.item.created":
                    item_id = msg.get("item").get("id")
                    if item_id:
                        self._item_timestamps[item_id] = tzutcnow()
                case "response.created":
                    response_id = msg["response"]["id"]
                    self._current_response_id = response_id
                    self._response_generations[response_id] = self._turn_generation
                    trace = self._turn()
                    trace.response_created_perf = time.perf_counter()
                    self._audit_record(
                        "assistant_response_created",
                        response_id=response_id,
                    )
                case "response.done":
                    response_id = msg["response"]["id"]
                    if self._current_response_id == response_id:
                        self._current_response_id = None
                    generation = self._response_generations.pop(
                        response_id, self._turn_generation
                    )
                    has_tool_call = await self._enqueue_tool_calls(msg["response"])
                    self._audit_record(
                        "assistant_response_done",
                        generation=generation,
                        response_id=response_id,
                        response_status=msg["response"].get("status"),
                        has_tool_call=has_tool_call,
                        usage=msg["response"].get("usage"),
                    )
                    if not has_tool_call:
                        self._complete_turn(
                            generation,
                            usage=msg["response"].get("usage"),
                        )
                case "response.output_audio.delta":
                    item_id = msg["item_id"]
                    generation = self._item_generations.setdefault(
                        item_id, self._turn_generation
                    )
                    trace = self._turn(generation)
                    if trace.first_audio_perf is None:
                        trace.first_audio_perf = time.perf_counter()
                        baseline = trace.speech_stopped_perf or trace.started_perf
                        self._audit_record(
                            "assistant_first_audio",
                            generation=generation,
                            item_id=item_id,
                            time_to_first_audio_ms=self._elapsed_ms(
                                baseline, trace.first_audio_perf
                            ),
                            post_retrieval_to_first_audio_ms=self._elapsed_ms(
                                trace.retrieval_done_perf, trace.first_audio_perf
                            ),
                        )
                    chunk = await run_in_thread(self._decode_chunk, msg["delta"])

                    if not conv_items_started_by_id.get(item_id):
                        conv_items_started_by_id[item_id] = True
                        await bus.publish(SegmentGenerationStarted(segment_id=item_id))

                    await bus.publish(SegmentChunkGenerated(segment_id=item_id, buffer=chunk))
                case "response.output_audio.done":
                    item_id = msg["item_id"]
                    if conv_items_started_by_id.get(item_id):
                        await bus.publish(SegmentGenerationCompleted(segment_id=item_id))
                        del conv_items_started_by_id[item_id]
                case "input_audio_buffer.speech_started":
                    _LOGGER.info("Speech interrupted")
                    self._turn_generation += 1
                    trace = self._turn()
                    trace.speech_started_perf = time.perf_counter()
                    self._audit_record("user_speech_started")
                    await bus.publish(DiscardAvatarSpeechBuffer())
                case "input_audio_buffer.speech_stopped":
                    trace = self._turn()
                    trace.speech_stopped_perf = time.perf_counter()
                    self._audit_record(
                        "user_speech_stopped",
                        speech_duration_ms=self._elapsed_ms(
                            trace.speech_started_perf, trace.speech_stopped_perf
                        ),
                    )
                case "conversation.item.input_audio_transcription.completed":
                    item_id = msg.get("item_id")
                    timestamp = self._item_timestamps.get(item_id, tzutcnow())
                    trace = self._turn()
                    trace.input_transcript = msg["transcript"]
                    trace.transcription_done_perf = time.perf_counter()
                    self._audit_record(
                        "user_input_completed",
                        input_mode="voice",
                        item_id=item_id,
                        input_transcript=trace.input_transcript,
                        transcription_after_speech_ms=self._elapsed_ms(
                            trace.speech_stopped_perf, trace.transcription_done_perf
                        ),
                    )
                    await self._send_event(
                        bus,
                        InputAudioTranscription(transcript=msg["transcript"], timestamp=timestamp),
                    )
                case "response.output_audio_transcript.done":
                    item_id = msg.get("item_id")
                    generation = self._item_generations.get(
                        str(item_id), self._turn_generation
                    )
                    trace = self._turn(generation)
                    trace.final_output = msg["transcript"]
                    self._audit_record(
                        "assistant_transcript_completed",
                        generation=generation,
                        item_id=item_id,
                        final_output=trace.final_output,
                    )
                    timestamp = self._item_timestamps.get(item_id, tzutcnow())
                    await self._send_event(
                        bus,
                        ResponseAudioTranscription(
                            transcript=msg["transcript"], timestamp=timestamp
                        ),
                    )
                case _:
                    pass

    async def _handle_bus_events(self, bus: EventBus, ws: ClientConnection) -> None:
        """Handle events from the event bus."""
        async with bus.subscribe(
            TextEchoEnqueueText,
            UserSpeechReceived,
            Shutdown,
        ) as sub:
            bus.ready()
            async for event in sub:
                match event:
                    case TextEchoEnqueueText(phrase_id=pid, text=txt):
                        await self._send_user_text(ws, pid, txt)
                    case UserSpeechReceived(buffer=buf):
                        await self._send_speech(ws, buf)
                    case Shutdown():
                        await ws.close()
                        return

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        """Connect to WebSocket and process messages."""
        if self._audit is not None:
            await self._audit.start()
        try:
            async with self._connect() as ws:
                tasks = {
                    asyncio.create_task(
                        self._listener(bus.clone(), ws),
                        name="RealtimeApiClient._listener",
                    ),
                    asyncio.create_task(
                        self._handle_bus_events(bus.clone(), ws),
                        name="RealtimeApiClient._handle_bus_events",
                    ),
                }
                tool_worker = asyncio.create_task(
                    self._tool_worker(ws), name="RealtimeApiClient._tool_worker"
                )
                bus.ready()
                try:
                    done, _ = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()
                finally:
                    await ws.close()
                    for task in (*tasks, tool_worker):
                        task.cancel()
                    await asyncio.gather(*tasks, tool_worker, return_exceptions=True)
        finally:
            if self._rag_client is not None:
                await self._rag_client.aclose()
            if self._audit is not None:
                await self._audit.close()
