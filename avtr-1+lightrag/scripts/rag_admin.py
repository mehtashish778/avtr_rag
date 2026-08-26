#!/usr/bin/env python3
"""Ingest one corpus into the configured RAG service or run a safe smoke query."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from avaturn_live_streamer.conversation_engines.rag_client import (
    DEFAULT_BACKEND,
    RagBackend,
    RagClient,
    RagSettings,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "knowledge_base" / "documents"


class CliSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_file=ROOT / ".env",
        extra="ignore",
    )

    enabled: bool = True
    backend: RagBackend = DEFAULT_BACKEND
    base_url: str = ""
    api_key: str = ""
    dataset_ids: list[str] = Field(default_factory=list)
    mode: str = "naive"
    top_k: int = 3
    timeout_seconds: float = 5.0
    max_context_chars: int = 12000
    similarity_threshold: float = 0.0
    vector_similarity_weight: float = 1.0

    def rag_settings(self) -> RagSettings:
        return RagSettings.model_validate(self.model_dump())


def load_settings() -> RagSettings:
    return CliSettings().rag_settings()


def document_files(directory: Path) -> list[Path]:
    allowed = {".txt", ".md"}
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )


def headers(backend: str, api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    if backend == "lightrag":
        return {"X-API-Key": api_key}
    return {"Authorization": f"Bearer {api_key}"}


async def ingest_txtai(
    http: httpx.AsyncClient, files: list[Path], auth: dict[str, str]
) -> None:
    payload = [
        {
            "id": path.stem,
            "text": path.read_text(encoding="utf-8"),
            "tags": path.name,
        }
        for path in files
    ]
    response = await http.post("/add", json=payload, headers=auth)
    response.raise_for_status()
    response = await http.get("/index", headers=auth)
    response.raise_for_status()


async def ingest_lightrag(
    http: httpx.AsyncClient, files: list[Path], auth: dict[str, str]
) -> None:
    for path in files:
        response = await http.post(
            "/documents/text",
            headers=auth,
            json={
                "text": path.read_text(encoding="utf-8"),
                "file_source": path.name,
                "description": path.stem,
            },
        )
        response.raise_for_status()


async def ingest_ragflow(
    http: httpx.AsyncClient,
    files: list[Path],
    auth: dict[str, str],
    dataset_id: str,
) -> None:
    uploads = [
        ("file", (path.name, path.read_bytes(), "text/plain; charset=utf-8"))
        for path in files
    ]
    response = await http.post(
        f"/api/v1/datasets/{dataset_id}/documents",
        files=uploads,
        headers=auth,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError("RAGFlow document upload returned an application error")
    document_ids = [
        item["id"]
        for item in body.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not document_ids:
        raise RuntimeError("RAGFlow upload returned no document IDs")
    response = await http.post(
        f"/api/v1/datasets/{dataset_id}/chunks",
        json={"document_ids": document_ids},
        headers=auth,
    )
    response.raise_for_status()
    parse_body = response.json()
    if parse_body.get("code") != 0:
        raise RuntimeError("RAGFlow document parsing returned an application error")


async def ingest(directory: Path) -> int:
    settings = load_settings()
    files = document_files(directory)
    if not files:
        raise FileNotFoundError(
            f"No .txt or .md documents found under {directory}. "
            "Add the same approved corpus to every AVTR variant."
        )
    if settings.backend == "ragflow" and len(settings.dataset_ids) != 1:
        raise ValueError("RAGFlow ingestion requires exactly one RAG__DATASET_IDS entry")

    async with httpx.AsyncClient(
        base_url=settings.resolved_base_url,
        timeout=httpx.Timeout(120.0),
    ) as http:
        auth = headers(settings.backend, settings.api_key)
        if settings.backend == "txtai":
            await ingest_txtai(http, files, auth)
        elif settings.backend == "lightrag":
            await ingest_lightrag(http, files, auth)
        else:
            await ingest_ragflow(http, files, auth, settings.dataset_ids[0])

    print(
        json.dumps(
            {
                "status": "submitted",
                "backend": settings.backend,
                "documents": len(files),
                "directory": str(directory),
            }
        )
    )
    if settings.backend in {"lightrag", "ragflow"}:
        print("Wait for document processing to finish before evaluation.")
    return 0


async def smoke(query: str) -> int:
    settings = load_settings()
    client = RagClient(settings)
    try:
        result = await client.query(query)
    finally:
        await client.aclose()
    summary = {
        "status": result.get("status"),
        "backend": result.get("backend"),
        "latency_ms": result.get("latency_ms"),
        "context_chars": len(str(result.get("context", ""))),
        "references": len(result.get("references", [])),
        "reason": result.get("reason"),
    }
    print(json.dumps(summary))
    return 0 if result.get("status") in {"ok", "empty"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument(
        "--documents-dir",
        type=Path,
        default=DEFAULT_DOCUMENTS,
        help="Directory containing the approved .txt/.md corpus",
    )

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument(
        "--query",
        default="What information is available in the knowledge base?",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "ingest":
        return asyncio.run(ingest(args.documents_dir.resolve()))
    return asyncio.run(smoke(args.query))


if __name__ == "__main__":
    raise SystemExit(main())
