# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Non-blocking structured audit logs for RAG-backed voice turns."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AuditLogSettings(BaseModel):
    """Environment-backed settings nested under ``AUDIT__``."""

    enabled: bool = True
    directory: str = "logs/rag-audit"
    full_content: bool = True
    max_value_chars: int = Field(default=200000, ge=1000, le=2000000)


class RagAuditLogger:
    """Write one JSON object per event without blocking the realtime listener."""

    _SECRET_KEYS = ("api_key", "authorization", "credential", "secret", "token")

    def __init__(
        self,
        settings: AuditLogSettings,
        *,
        stream_id: str,
        backend: str,
    ) -> None:
        self.settings = settings
        self.stream_id = stream_id or "unknown-stream"
        self.backend = backend
        safe_stream = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.stream_id)[:80]
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = Path(settings.directory).expanduser() / (
            f"{backend}_{stamp}_{safe_stream}.jsonl"
        )
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self.settings.enabled or self._worker_task is not None:
            return
        await asyncio.to_thread(self._prepare_file)
        self._worker_task = asyncio.create_task(
            self._worker(), name="RagAuditLogger._worker"
        )
        self.record("session_started", log_path=str(self.path))

    def record(self, event: str, *, turn_id: str | None = None, **details: Any) -> None:
        if not self.settings.enabled:
            return
        payload: dict[str, Any] = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="microseconds"),
            "event": event,
            "stream_id": self.stream_id,
            "turn_id": turn_id,
            "backend": self.backend,
            **details,
        }
        self._queue.put_nowait(self._sanitize(payload))

    async def close(self) -> None:
        if not self.settings.enabled or self._worker_task is None:
            return
        self.record("session_ended")
        await self._queue.join()
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    def _prepare_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        self.path.touch(exist_ok=True)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    async def _worker(self) -> None:
        while True:
            payload = await self._queue.get()
            try:
                if payload is None:
                    return
                await asyncio.to_thread(self._append, payload)
            finally:
                self._queue.task_done()

    def _append(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()

    def _sanitize(self, value: Any, key: str = "") -> Any:
        if any(secret in key.lower() for secret in self._SECRET_KEYS):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {str(k): self._sanitize(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(item, key) for item in value]
        if isinstance(value, str):
            if self.settings.full_content or len(value) <= self.settings.max_value_chars:
                return value
            return value[: self.settings.max_value_chars] + "...[TRUNCATED]"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)
