#!/usr/bin/env python3
"""Bounded one-GPU migration smoke test for the canonical CIAI workspace."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import IO

import httpx
from dotenv import dotenv_values


ROOT = Path("/home/ashish.mehta/avtr_rag")
RUNTIME = Path("/l/users/ashish.mehta/avtr_rag-runtime")
LOG_ROOT = RUNTIME / "migration-smoke" / "logs"
LIGHTRAG = ROOT / "avtr-1+lightrag"
TXTAI = ROOT / "avtr-1+txtai"
JOB_ID = os.environ.get("SLURM_JOB_ID", "manual")


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout.strip()
    if output:
        print(output, flush=True)
    return output


def variant_env(project: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({key: value for key, value in dotenv_values(project / ".env").items() if value})
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{project / 'src'}{os.pathsep + existing if existing else ''}"
    return env


def start(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: IO[bytes],
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def wait_for_url(
    url: str,
    process: subprocess.Popen[bytes],
    log_path: Path,
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not attempted"
    with httpx.Client(timeout=3.0) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Process exited with {process.returncode} before {url} became healthy; "
                    f"see {log_path}"
                )
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}; see {log_path}")


def parse_smoke(output: str, backend: str) -> dict[str, object]:
    candidates = [line for line in output.splitlines() if line.lstrip().startswith("{")]
    if not candidates:
        raise RuntimeError(f"{backend} smoke command returned no JSON result")
    result = json.loads(candidates[-1])
    if result.get("status") != "ok":
        raise RuntimeError(f"{backend} retrieval status was {result.get('status')!r}: {result}")
    if int(result.get("context_chars", 0)) <= 0 or int(result.get("references", 0)) <= 0:
        raise RuntimeError(f"{backend} returned no grounded context or references: {result}")
    print(f"{backend.upper()}_RESULT={json.dumps(result, sort_keys=True)}", flush=True)
    return result


def check_layout() -> None:
    required = [
        ROOT / ".git",
        LIGHTRAG / ".env",
        LIGHTRAG / ".pixi",
        LIGHTRAG / ".rag-venv" / "bin" / "python",
        TXTAI / ".env",
        TXTAI / ".pixi",
        TXTAI / ".rag-venv" / "bin" / "python",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing migrated paths: {missing}")


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    check_layout()
    print("MIGRATION_SMOKE_START", flush=True)
    print(
        f"host={os.uname().nodename} job={JOB_ID} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        flush=True,
    )

    light_env = variant_env(LIGHTRAG)
    txtai_env = variant_env(TXTAI)
    txtai_env["CONFIG"] = str(TXTAI / "service-configs" / "txtai-openai.yml")

    run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
        cwd=ROOT,
        env=light_env,
    )
    run(
        [
            "pixi",
            "run",
            "-e",
            "default",
            "python",
            "-c",
            (
                "import pathlib,torch,avtr1_renderer;"
                f"root=pathlib.Path({str(LIGHTRAG)!r});"
                "module=pathlib.Path(avtr1_renderer.__file__).resolve();"
                "print(f'CUDA_OK={torch.cuda.is_available()} GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} RENDERER_MODULE={module}');"
                "assert torch.cuda.is_available();assert module.is_relative_to(root)"
            ),
        ],
        cwd=LIGHTRAG,
        env=light_env,
    )
    run(
        [
            "pixi",
            "run",
            "-e",
            "streamer",
            "python",
            "-c",
            (
                "import pathlib,avaturn_live_streamer;"
                f"root=pathlib.Path({str(LIGHTRAG)!r});"
                "module=pathlib.Path(avaturn_live_streamer.__file__).resolve();"
                "print(f'STREAMER_MODULE={module}');assert module.is_relative_to(root)"
            ),
        ],
        cwd=LIGHTRAG,
        env=light_env,
    )

    light_process: subprocess.Popen[bytes] | None = None
    orchestrator_process: subprocess.Popen[bytes] | None = None
    txtai_process: subprocess.Popen[bytes] | None = None

    with ExitStack() as stack:
        light_log_path = LOG_ROOT / f"lightrag_{JOB_ID}.log"
        light_log = stack.enter_context(light_log_path.open("wb"))
        orchestrator_log_path = LOG_ROOT / f"avtr_orchestrator_{JOB_ID}.log"
        orchestrator_log = stack.enter_context(orchestrator_log_path.open("wb"))
        txtai_log_path = LOG_ROOT / f"txtai_{JOB_ID}.log"
        txtai_log = stack.enter_context(txtai_log_path.open("wb"))
        try:
            light_process = start(
                [str(LIGHTRAG / ".rag-venv" / "bin" / "lightrag-server")],
                cwd=LIGHTRAG,
                env=light_env,
                log=light_log,
            )
            wait_for_url(
                "http://127.0.0.1:9621/health",
                light_process,
                light_log_path,
                timeout_seconds=240,
            )
            light_output = run(
                [
                    "pixi",
                    "run",
                    "-e",
                    "streamer",
                    "python",
                    "scripts/rag_admin.py",
                    "smoke",
                    "--query",
                    "What breast changes should I report to a doctor?",
                ],
                cwd=LIGHTRAG,
                env=light_env,
            )
            parse_smoke(light_output, "lightrag")

            orchestrator_process = start(
                ["pixi", "run", "-e", "streamer", "python", "scripts/run_local_stream.py"],
                cwd=LIGHTRAG,
                env=light_env,
                log=orchestrator_log,
            )
            wait_for_url(
                "http://127.0.0.1:8000/health",
                orchestrator_process,
                orchestrator_log_path,
                timeout_seconds=360,
            )
            wait_for_url(
                "http://127.0.0.1:7860/",
                orchestrator_process,
                orchestrator_log_path,
                timeout_seconds=60,
            )
            print("AVTR_RENDERER_AND_STREAMER_OK", flush=True)
            stop(orchestrator_process)
            orchestrator_process = None
            stop(light_process)
            light_process = None

            run(
                [
                    "pixi",
                    "run",
                    "-e",
                    "streamer",
                    "python",
                    "-c",
                    (
                        "import pathlib,avaturn_live_streamer;"
                        f"root=pathlib.Path({str(TXTAI)!r});"
                        "module=pathlib.Path(avaturn_live_streamer.__file__).resolve();"
                        "print(f'TXTAI_STREAMER_MODULE={module}');assert module.is_relative_to(root)"
                    ),
                ],
                cwd=TXTAI,
                env=txtai_env,
            )
            txtai_process = start(
                [
                    str(TXTAI / ".rag-venv" / "bin" / "python"),
                    "-m",
                    "uvicorn",
                    "txtai.api:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8001",
                    "--no-access-log",
                ],
                cwd=TXTAI,
                env=txtai_env,
                log=txtai_log,
            )
            wait_for_url(
                "http://127.0.0.1:8001/openapi.json",
                txtai_process,
                txtai_log_path,
                timeout_seconds=240,
            )
            txtai_output = run(
                [
                    "pixi",
                    "run",
                    "-e",
                    "streamer",
                    "python",
                    "scripts/rag_admin.py",
                    "smoke",
                    "--query",
                    "What breast changes should I report to a doctor?",
                ],
                cwd=TXTAI,
                env=txtai_env,
            )
            parse_smoke(txtai_output, "txtai")
            stop(txtai_process)
            txtai_process = None

            run(
                [
                    "pixi",
                    "run",
                    "-e",
                    "streamer",
                    "python",
                    "-c",
                    (
                        "from avaturn_live_streamer.conversation_engines.rag_client import RagClient,RagSettings;"
                        "cfg=RagSettings(enabled=True,backend='ragflow',base_url='http://127.0.0.1:9380');"
                        "RagClient(cfg);print('RAGFLOW_INTEGRATION_CODE_OK service_runtime=deferred')"
                    ),
                ],
                cwd=TXTAI,
                env=txtai_env,
            )
        finally:
            stop(txtai_process)
            stop(orchestrator_process)
            stop(light_process)

    print("MIGRATION_GPU_SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
