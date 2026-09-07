#!/usr/bin/env python3
"""Matched scalability benchmark for the txtai and LightRAG retrieval APIs."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

Backend = Literal["txtai", "lightrag"]


@dataclass(frozen=True)
class Question:
    test_id: str
    text: str
    expected_doc_ids: tuple[str, ...]


@dataclass
class Observation:
    backend: str
    concurrency: int
    sequence: int
    test_id: str
    latency_ms: float
    http_status: int | None
    success: bool
    retrieval_hit: bool
    response_bytes: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("txtai", "lightrag"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,4,8,16,32")
    parser.add_argument("--requests-per-level", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--service-match", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            questions.append(
                Question(
                    test_id=str(item["id"]),
                    text=str(item["question"]),
                    expected_doc_ids=tuple(str(x) for x in item["expected_doc_ids"]),
                )
            )
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return round(ordered[index], 2)


def matching_pids(pattern: str) -> list[int]:
    if not pattern:
        return []
    try:
        output = subprocess.check_output(
            ["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [int(value) for value in output.split() if value.isdigit() and int(value) != os.getpid()]


def process_snapshot(pattern: str) -> tuple[float, int, int]:
    cpu_seconds = 0.0
    rss_bytes = 0
    process_count = 0
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    for pid in matching_pids(pattern):
        try:
            stat_fields = Path(f"/proc/{pid}/stat").read_text().split()
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        cpu_seconds += (int(stat_fields[13]) + int(stat_fields[14])) / clock_ticks
        for line in status:
            if line.startswith("VmRSS:"):
                rss_bytes += int(line.split()[1]) * 1024
                break
        process_count += 1
    return cpu_seconds, rss_bytes, process_count


async def request_once(
    client: httpx.AsyncClient,
    backend: Backend,
    question: Question,
    top_k: int,
) -> tuple[int, bytes]:
    if backend == "txtai":
        response = await client.get(
            "/search", params={"query": question.text, "limit": top_k}
        )
    else:
        response = await client.post(
            "/query",
            json={
                "query": question.text,
                "mode": "naive",
                "only_need_context": True,
                "top_k": top_k,
                "chunk_top_k": top_k,
                "enable_rerank": False,
            },
        )
    body = await response.aread()
    response.raise_for_status()
    return response.status_code, body


async def measured_request(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    backend: Backend,
    concurrency: int,
    sequence: int,
    question: Question,
    top_k: int,
) -> Observation:
    async with semaphore:
        started = time.perf_counter()
        status: int | None = None
        body = b""
        try:
            status, body = await request_once(client, backend, question, top_k)
            decoded = body.decode("utf-8", errors="replace")
            hit = any(doc_id in decoded for doc_id in question.expected_doc_ids)
            return Observation(
                backend=backend,
                concurrency=concurrency,
                sequence=sequence,
                test_id=question.test_id,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                http_status=status,
                success=True,
                retrieval_hit=hit,
                response_bytes=len(body),
            )
        except Exception as exc:
            return Observation(
                backend=backend,
                concurrency=concurrency,
                sequence=sequence,
                test_id=question.test_id,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                http_status=status,
                success=False,
                retrieval_hit=False,
                response_bytes=len(body),
                error=f"{type(exc).__name__}: {exc}",
            )


async def run_level(
    *,
    client: httpx.AsyncClient,
    backend: Backend,
    concurrency: int,
    request_count: int,
    questions: list[Question],
    top_k: int,
    service_match: str,
) -> tuple[dict[str, Any], list[Observation]]:
    semaphore = asyncio.Semaphore(concurrency)
    cpu_before, _, processes_before = process_snapshot(service_match)
    started = time.perf_counter()
    tasks = [
        measured_request(
            semaphore,
            client,
            backend,
            concurrency,
            sequence,
            questions[sequence % len(questions)],
            top_k,
        )
        for sequence in range(request_count)
    ]
    observations = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    cpu_after, rss_after, processes_after = process_snapshot(service_match)

    successes = [item for item in observations if item.success]
    latencies = [item.latency_ms for item in successes]
    errors: dict[str, int] = {}
    for item in observations:
        if item.error:
            errors[item.error] = errors.get(item.error, 0) + 1
    summary: dict[str, Any] = {
        "concurrency": concurrency,
        "requests": len(observations),
        "successes": len(successes),
        "errors": len(observations) - len(successes),
        "error_rate": round((len(observations) - len(successes)) / len(observations), 4),
        "retrieval_hit_rate": round(
            sum(item.retrieval_hit for item in successes) / len(successes), 4
        )
        if successes
        else 0.0,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_requests_per_second": round(len(successes) / elapsed, 3),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": round(max(latencies), 2) if latencies else None,
        },
        "response_bytes_mean": round(
            sum(item.response_bytes for item in successes) / len(successes), 1
        )
        if successes
        else None,
        "service_resources": {
            "process_match": service_match,
            "process_count_before": processes_before,
            "process_count_after": processes_after,
            "cpu_core_percent_during_level": round(
                max(0.0, cpu_after - cpu_before) / elapsed * 100, 2
            ),
            "rss_mib_after": round(rss_after / (1024 * 1024), 2),
        },
        "error_counts": errors,
    }
    return summary, observations


async def main_async(args: argparse.Namespace) -> int:
    backend: Backend = args.backend
    api_key = args.api_key or os.environ.get("RAG__API_KEY", "")
    levels = [int(item) for item in args.concurrency.split(",") if item.strip()]
    if not levels or min(levels) < 1:
        raise ValueError("Concurrency values must be positive integers")
    if args.requests_per_level < max(levels):
        raise ValueError("requests-per-level must be at least the maximum concurrency")

    questions = load_questions(args.questions)
    headers: dict[str, str] = {}
    if api_key:
        header = "X-API-Key" if backend == "lightrag" else "Authorization"
        headers[header] = api_key if backend == "lightrag" else f"Bearer {api_key}"
    limits = httpx.Limits(
        max_connections=max(levels) + 8,
        max_keepalive_connections=max(levels) + 8,
    )
    timeout = httpx.Timeout(args.timeout_seconds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observation_path = args.output_dir / "observations.jsonl"
    summaries: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
        limits=limits,
    ) as client:
        for index in range(args.warmup):
            await request_once(client, backend, questions[index % len(questions)], args.top_k)
        with observation_path.open("w", encoding="utf-8") as observations_file:
            for concurrency in levels:
                summary, observations = await run_level(
                    client=client,
                    backend=backend,
                    concurrency=concurrency,
                    request_count=args.requests_per_level,
                    questions=questions,
                    top_k=args.top_k,
                    service_match=args.service_match,
                )
                summaries.append(summary)
                for observation in observations:
                    observations_file.write(json.dumps(asdict(observation), ensure_ascii=False))
                    observations_file.write("\n")
                observations_file.flush()
                print(json.dumps({"backend": backend, **summary}), flush=True)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": backend,
        "base_url": args.base_url,
        "host": platform.node(),
        "python": platform.python_version(),
        "questions": len(questions),
        "top_k": args.top_k,
        "warmup_requests": args.warmup,
        "requests_per_level": args.requests_per_level,
        "concurrency_levels": levels,
        "percentile_method": "nearest-rank over successful request wall latency",
        "levels": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0 if all(level["errors"] == 0 for level in summaries) else 2


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
