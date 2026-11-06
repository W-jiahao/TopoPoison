from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .build import build_datasets
from .config import DEFAULT_CONFIG, PACKAGE_ROOT, load_config, select_datasets
from .doctor import check_environment
from .evaluate import CONDITIONS, SYSTEMS, evaluate_datasets
from .io import read_json, sha256_file, write_json_atomic


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and dash")
    return value


def selected_systems(values: Iterable[str] | None) -> list[str]:
    systems = list(values) if values else list(SYSTEMS)
    unknown = [value for value in systems if value not in SYSTEMS]
    if unknown:
        raise ValueError(f"Unknown systems: {', '.join(unknown)}")
    return systems


def redact_api_keys(command: list[str]) -> list[str]:
    displayed = list(command)
    for index, value in enumerate(displayed):
        if value == "--api-key" and index + 1 < len(displayed):
            displayed[index + 1] = "<redacted>"
        elif value.startswith("--api-key="):
            displayed[index] = "--api-key=<redacted>"
    return displayed


def contract_mismatches(
    existing: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    return [key for key, value in expected.items() if existing.get(key) != value]


def call_runner(
    python: Path,
    module: str,
    dataset: str,
    condition: str,
    run_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    config_path: Path,
    limit: int | None,
    resume: bool,
    log_path: Path,
) -> None:
    command = [
        str(python),
        "-m",
        module,
        "--dataset",
        dataset,
        "--condition",
        condition,
        "--run-dir",
        str(run_dir),
        "--base-url",
        base_url,
        "--model",
        model,
        "--config",
        str(config_path),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if resume:
        command.append("--resume")
    displayed_command = redact_api_keys(command)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    env["VLLM_API_KEY"] = api_key
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(displayed_command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PACKAGE_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Runner failed; inspect {log_path}")


def reproduce(
    names: Iterable[str] | None,
    systems: Iterable[str] | None,
    run_id: str | None,
    graph_python: Path,
    hippo_python: Path,
    base_url: str,
    api_key: str,
    model: str,
    limit: int | None,
    resume: bool,
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[Path, list[dict[str, Any]]]:
    config = load_config(config_path)
    datasets = select_datasets(config, names)
    selected = selected_systems(systems)
    identifier = validate_run_id(run_id or default_run_id())
    run_dir = PACKAGE_ROOT / "runs" / identifier
    if resume and run_id is None:
        raise ValueError("--resume requires an explicit --run-id")
    if resume and not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose another run-id or use --resume."
        )

    contract = {
        "method": "TopoPoison",
        "run_id": identifier,
        "datasets": datasets,
        "conditions": list(CONDITIONS),
        "systems": selected,
        "query_limit": limit,
        "model": model,
        "vllm_base_url": base_url.rstrip("/"),
        "config_sha256": sha256_file(config_path),
    }
    existing_manifest: dict[str, Any] | None = None
    if resume:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
        existing_manifest = read_json(manifest_path)
        mismatches = contract_mismatches(existing_manifest, contract)
        if mismatches:
            raise ValueError(
                "Resume arguments differ from the original run: " + ", ".join(mismatches)
            )

    environment = check_environment(
        selected, graph_python, hippo_python, base_url, api_key, model
    )
    build_datasets(datasets)
    run_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat()
    if existing_manifest is None:
        manifest = {
            **contract,
            "created_at": now,
            "fresh_run": True,
            "environment": environment,
            "command": redact_api_keys(sys.argv),
            "resume_history": [],
        }
    else:
        manifest = existing_manifest
        manifest.setdefault("resume_history", []).append(
            {
                "resumed_at": now,
                "environment": environment,
                "command": redact_api_keys(sys.argv),
            }
        )
    write_json_atomic(run_dir / "manifest.json", manifest)

    for system in selected:
        if system == "GraphRAG":
            python = graph_python
            module = "topopoison.graphrag_runner"
        else:
            python = hippo_python
            module = "topopoison.hipporag_runner"
        for dataset in datasets:
            for condition in CONDITIONS:
                call_runner(
                    python=python,
                    module=module,
                    dataset=dataset,
                    condition=condition,
                    run_dir=run_dir,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    config_path=config_path,
                    limit=limit,
                    resume=resume,
                    log_path=run_dir
                    / "logs"
                    / "orchestrator"
                    / system
                    / dataset
                    / f"{condition}.log",
                )

    rows = evaluate_datasets(
        run_dir=run_dir,
        names=datasets,
        systems=selected,
        limit=limit,
        config_path=config_path,
    )
    return run_dir, rows
