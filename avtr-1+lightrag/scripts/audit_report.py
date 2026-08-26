#!/usr/bin/env python3
"""Summarize AVTR RAG audit JSONL files and score known evaluation questions."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "logs" / "rag-audit"
DEFAULT_QUESTIONS = ROOT / "knowledge_base" / "evaluation" / "breast_cancer_questions.jsonl"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[index], 2)


def find_question(
    transcript: str, questions: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    candidate = normalize(transcript)
    best: dict[str, Any] | None = None
    best_score = 0.0
    for question in questions:
        score = SequenceMatcher(None, candidate, normalize(str(question["question"]))).ratio()
        if score > best_score:
            best, best_score = question, score
    return (best, best_score) if best_score >= 0.72 else (None, best_score)


def build_report(log_files: list[Path], questions_path: Path) -> dict[str, Any]:
    questions = load_jsonl(questions_path)
    turns = [
        row
        for path in log_files
        for row in load_jsonl(path)
        if row.get("event") == "turn_completed"
    ]
    scored: list[dict[str, Any]] = []
    total_times: list[float] = []
    retrieval_times: list[float] = []
    first_audio_times: list[float] = []

    for turn in turns:
        timings = turn.get("timings") or {}
        for target, key in (
            (total_times, "total_turn_ms"),
            (retrieval_times, "retrieval_ms"),
            (first_audio_times, "time_to_first_audio_ms"),
        ):
            value = timings.get(key)
            if isinstance(value, (int, float)):
                target.append(float(value))

        expected, similarity = find_question(str(turn.get("input_transcript", "")), questions)
        if expected is None:
            continue
        retrieval_text = json.dumps(turn.get("retrieval", {}), ensure_ascii=False).lower()
        final_text = normalize(str(turn.get("final_output", "")))
        expected_docs = [str(item) for item in expected.get("expected_doc_ids", [])]
        expected_terms = [normalize(str(item)) for item in expected.get("expected_terms", [])]
        doc_hits = [doc_id for doc_id in expected_docs if doc_id.lower() in retrieval_text]
        term_hits = [term for term in expected_terms if term and term in final_text]
        scored.append(
            {
                "test_id": expected.get("id"),
                "turn_id": turn.get("turn_id"),
                "question_similarity": round(similarity, 3),
                "retrieval_hit": bool(doc_hits),
                "expected_doc_ids": expected_docs,
                "matched_doc_ids": doc_hits,
                "answer_term_recall": (
                    round(len(term_hits) / len(expected_terms), 3) if expected_terms else None
                ),
                "matched_terms": term_hits,
                "timings": timings,
            }
        )

    term_recalls = [
        float(row["answer_term_recall"])
        for row in scored
        if row["answer_term_recall"] is not None
    ]
    return {
        "log_files": [str(path) for path in log_files],
        "turns": len(turns),
        "scored_turns": len(scored),
        "retrieval_hit_rate": (
            round(sum(bool(row["retrieval_hit"]) for row in scored) / len(scored), 3)
            if scored
            else None
        ),
        "mean_answer_term_recall": (
            round(statistics.fmean(term_recalls), 3) if term_recalls else None
        ),
        "latency_ms": {
            "retrieval_p50": percentile(retrieval_times, 0.50),
            "retrieval_p95": percentile(retrieval_times, 0.95),
            "first_audio_p50": percentile(first_audio_times, 0.50),
            "first_audio_p95": percentile(first_audio_times, 0.95),
            "total_turn_p50": percentile(total_times, 0.50),
            "total_turn_p95": percentile(total_times, 0.95),
        },
        "scored_results": scored,
        "notes": [
            "Retrieval hit checks whether an expected document ID appears in logged evidence.",
            "Answer-term recall is a deterministic quality proxy, not a clinical correctness judgment.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_files = sorted(args.log_dir.glob("*.jsonl"))
    if not log_files:
        raise FileNotFoundError(f"No audit JSONL files found under {args.log_dir}")
    report = build_report(log_files, args.questions)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
