#!/usr/bin/env python3
"""Evaluate ranked retrieval accuracy for txtai or LightRAG."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

Backend = Literal["txtai", "lightrag"]
DOC_ID_RE = re.compile(r"BC-\d{3}", re.IGNORECASE)
KS = (1, 3, 5)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    base_id: str
    category: str
    difficulty: str
    question: str
    expected_doc_ids: tuple[str, ...]


@dataclass
class Observation:
    backend: str
    case_id: str
    base_id: str
    category: str
    difficulty: str
    repeat: int
    expected_doc_ids: list[str]
    ranked_doc_ids: list[str]
    scores: list[float | None]
    latency_ms: float
    response_bytes: int
    http_status: int | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("txtai", "lightrag"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            case_id = str(item["id"])
            if case_id in seen:
                raise ValueError(f"Duplicate id {case_id!r} at line {line_number}")
            expected = tuple(str(value).upper() for value in item["expected_doc_ids"])
            if not expected:
                raise ValueError(f"Case {case_id!r} has no expected documents")
            seen.add(case_id)
            cases.append(
                EvalCase(
                    case_id=case_id,
                    base_id=str(item.get("base_id", case_id)),
                    category=str(item.get("category", "unspecified")),
                    difficulty=str(item.get("difficulty", "unspecified")),
                    question=str(item["question"]),
                    expected_doc_ids=expected,
                )
            )
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def doc_id(value: Any) -> str | None:
    match = DOC_ID_RE.search(str(value or ""))
    return match.group(0).upper() if match else None


def unique_docs(values: list[str], top_k: int) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
        if len(result) >= top_k:
            break
    return result


def extract_txtai(payload: Any, top_k: int) -> tuple[list[str], list[float | None]]:
    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("txtai response does not contain a result list")
    ranked: list[str] = []
    score_by_doc: dict[str, float | None] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        found = None
        for key in ("tags", "id", "source", "text", "content"):
            found = doc_id(row.get(key))
            if found:
                break
        if not found or found in score_by_doc:
            continue
        raw_score = row.get("score")
        score_by_doc[found] = float(raw_score) if isinstance(raw_score, (int, float)) else None
        ranked.append(found)
        if len(ranked) >= top_k:
            break
    return ranked, [score_by_doc[value] for value in ranked]


def extract_lightrag(payload: Any, top_k: int) -> tuple[list[str], list[float | None]]:
    if not isinstance(payload, dict):
        raise ValueError("LightRAG response is not an object")
    ranked: list[str] = []
    scores: list[float | None] = []
    references = payload.get("references") or []
    if isinstance(references, list):
        for reference in references:
            if not isinstance(reference, dict):
                continue
            found = None
            for key in ("file_path", "source", "document_id", "id", "content"):
                found = doc_id(reference.get(key))
                if found:
                    break
            if not found or found in ranked:
                continue
            raw_score = reference.get("score") or reference.get("similarity")
            ranked.append(found)
            scores.append(float(raw_score) if isinstance(raw_score, (int, float)) else None)
            if len(ranked) >= top_k:
                break
    if len(ranked) < top_k:
        context = payload.get("response") or payload.get("context") or ""
        for found in DOC_ID_RE.findall(str(context)):
            normalized = found.upper()
            if normalized not in ranked:
                ranked.append(normalized)
                scores.append(None)
            if len(ranked) >= top_k:
                break
    return ranked, scores


async def request_once(
    client: httpx.AsyncClient,
    backend: Backend,
    case: EvalCase,
    top_k: int,
) -> tuple[int, bytes]:
    if backend == "txtai":
        response = await client.get(
            "/search", params={"query": case.question, "limit": top_k}
        )
    else:
        response = await client.post(
            "/query",
            json={
                "query": case.question,
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


async def evaluate_one(
    *,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    backend: Backend,
    case: EvalCase,
    repeat: int,
    top_k: int,
) -> Observation:
    async with semaphore:
        started = time.perf_counter()
        status: int | None = None
        body = b""
        try:
            status, body = await request_once(client, backend, case, top_k)
            payload = json.loads(body)
            if backend == "txtai":
                ranked, scores = extract_txtai(payload, top_k)
            else:
                ranked, scores = extract_lightrag(payload, top_k)
            error = None if ranked else "empty_ranked_documents"
        except Exception as exc:
            ranked, scores = [], []
            error = f"{type(exc).__name__}: {exc}"
        return Observation(
            backend=backend,
            case_id=case.case_id,
            base_id=case.base_id,
            category=case.category,
            difficulty=case.difficulty,
            repeat=repeat,
            expected_doc_ids=list(case.expected_doc_ids),
            ranked_doc_ids=ranked,
            scores=scores,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            response_bytes=len(body),
            http_status=status,
            error=error,
        )


def dcg(relevances: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))


def observation_metrics(observation: Observation) -> dict[str, float]:
    expected = set(observation.expected_doc_ids)
    ranked = observation.ranked_doc_ids
    metrics: dict[str, float] = {}
    for k in KS:
        returned = ranked[:k]
        matches = len(expected.intersection(returned))
        metrics[f"hit_at_{k}"] = float(matches > 0)
        metrics[f"precision_at_{k}"] = matches / k
        metrics[f"recall_at_{k}"] = matches / len(expected)
        ideal = [1] * min(len(expected), k)
        actual = [int(value in expected) for value in returned]
        metrics[f"ndcg_at_{k}"] = dcg(actual) / dcg(ideal) if ideal else 0.0
        metrics[f"full_recall_at_{k}"] = float(matches == len(expected))
    first_rank = next(
        (index for index, value in enumerate(ranked[:5], start=1) if value in expected),
        None,
    )
    metrics["reciprocal_rank_at_5"] = 1.0 / first_rank if first_rank else 0.0
    metrics["first_relevant_rank"] = float(first_rank or 0)
    return metrics


def aggregate(observations: list[Observation]) -> dict[str, Any]:
    successful = [item for item in observations if item.error is None]
    metric_rows = [observation_metrics(item) for item in successful]
    metric_names = [
        *(f"hit_at_{k}" for k in KS),
        *(f"precision_at_{k}" for k in KS),
        *(f"recall_at_{k}" for k in KS),
        *(f"ndcg_at_{k}" for k in KS),
        *(f"full_recall_at_{k}" for k in KS),
        "reciprocal_rank_at_5",
    ]
    result: dict[str, Any] = {
        "observations": len(observations),
        "successful": len(successful),
        "errors": len(observations) - len(successful),
    }
    for name in metric_names:
        result[name] = round(
            sum(row[name] for row in metric_rows) / len(metric_rows), 4
        ) if metric_rows else 0.0
    latencies = sorted(item.latency_ms for item in successful)
    result["latency_ms"] = {
        "mean": round(statistics.fmean(latencies), 2) if latencies else None,
        "p50": nearest_rank(latencies, 0.50),
        "p95": nearest_rank(latencies, 0.95),
        "max": round(max(latencies), 2) if latencies else None,
    }
    return result


def nearest_rank(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return round(values[max(0, math.ceil(probability * len(values)) - 1)], 2)


def stability(observations: list[Observation]) -> dict[str, float]:
    by_case: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.error is None:
            by_case[observation.case_id].append(observation)
    exact = 0
    top1 = 0
    evaluated = 0
    for rows in by_case.values():
        if len(rows) < 2:
            continue
        evaluated += 1
        rankings = [tuple(row.ranked_doc_ids) for row in rows]
        top_docs = [ranking[0] if ranking else None for ranking in rankings]
        exact += int(len(set(rankings)) == 1)
        top1 += int(len(set(top_docs)) == 1)
    return {
        "queries_with_repeats": evaluated,
        "exact_top5_stability": round(exact / evaluated, 4) if evaluated else 0.0,
        "top1_stability": round(top1 / evaluated, 4) if evaluated else 0.0,
    }


def failures(observations: list[Observation]) -> list[dict[str, Any]]:
    first_by_case: dict[str, Observation] = {}
    for observation in sorted(observations, key=lambda item: item.repeat):
        first_by_case.setdefault(observation.case_id, observation)
    rows: list[dict[str, Any]] = []
    for observation in first_by_case.values():
        metrics = observation_metrics(observation) if observation.error is None else {}
        if observation.error is None and metrics.get("full_recall_at_3") == 1.0:
            continue
        rows.append(
            {
                "case_id": observation.case_id,
                "base_id": observation.base_id,
                "category": observation.category,
                "difficulty": observation.difficulty,
                "expected_doc_ids": observation.expected_doc_ids,
                "ranked_doc_ids": observation.ranked_doc_ids,
                "hit_at_3": metrics.get("hit_at_3", 0.0),
                "recall_at_3": metrics.get("recall_at_3", 0.0),
                "error": observation.error,
            }
        )
    return sorted(rows, key=lambda item: (item["category"], item["case_id"]))


async def main_async(args: argparse.Namespace) -> int:
    backend: Backend = args.backend
    cases = load_cases(args.dataset)
    if args.top_k < 5:
        raise ValueError("top-k must be at least 5 to calculate the configured metrics")
    if args.repeats < 1 or args.concurrency < 1:
        raise ValueError("repeats and concurrency must be positive")

    api_key = args.api_key or os.environ.get("RAG__API_KEY", "")
    headers: dict[str, str] = {}
    if api_key:
        header = "X-API-Key" if backend == "lightrag" else "Authorization"
        headers[header] = api_key if backend == "lightrag" else f"Bearer {api_key}"

    jobs = [(case, repeat) for repeat in range(args.repeats) for case in cases]
    random.Random(args.seed).shuffle(jobs)
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency + 4,
        max_keepalive_connections=args.concurrency + 4,
    )
    started = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(args.timeout_seconds),
        limits=limits,
    ) as client:
        observations = await asyncio.gather(
            *[
                evaluate_one(
                    semaphore=semaphore,
                    client=client,
                    backend=backend,
                    case=case,
                    repeat=repeat,
                    top_k=args.top_k,
                )
                for case, repeat in jobs
            ]
        )
    elapsed = time.perf_counter() - started

    by_category = {
        category: aggregate([item for item in observations if item.category == category])
        for category in sorted({item.category for item in observations})
    }
    by_difficulty = {
        difficulty: aggregate(
            [item for item in observations if item.difficulty == difficulty]
        )
        for difficulty in sorted({item.difficulty for item in observations})
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": backend,
        "dataset": str(args.dataset),
        "unique_queries": len(cases),
        "categories": {
            category: sum(case.category == category for case in cases)
            for category in sorted({case.category for case in cases})
        },
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "top_k_requested": args.top_k,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_requests_per_second": round(len(observations) / elapsed, 3),
        "overall": aggregate(observations),
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "stability": stability(observations),
        "full_recall_at_3_failures": failures(observations),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(asdict(observation), ensure_ascii=False))
            handle.write("\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["overall"]["errors"] == 0 else 2


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
