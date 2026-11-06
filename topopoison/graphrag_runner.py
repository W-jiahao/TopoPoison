from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import DATASET_NAMES, DEFAULT_CONFIG, PACKAGE_ROOT, load_config
from .evaluate import output_file, record_id
from .io import iter_jsonl, write_json_atomic


PROMPT_ROOT = Path(__file__).resolve().parent / "prompts" / "graphrag"


def load_queries(path: Path, limit: int | None) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    for row in iter_jsonl(path):
        qid = record_id(row)
        text = str(row.get("text") or row.get("question") or "").strip()
        if qid is not None and text:
            queries.append((qid, text))
        if limit is not None and len(queries) >= limit:
            break
    return queries


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        qid
        for row in iter_jsonl(path)
        if (qid := record_id(row)) is not None
        and str(row.get("output") or "").strip()
        and not row.get("error")
        and not row.get("timed_out")
    }


def settings_text(
    dataset_spec: dict[str, Any],
    global_spec: dict[str, Any],
    model: str,
    base_url: str,
) -> str:
    graph = dataset_spec["graphrag"]
    entity_types = ", ".join(graph["entity_types"])
    return f'''models:
  default_chat_model:
    type: chat
    model_provider: openai
    auth_type: api_key
    api_key: ${{GRAPHRAG_API_KEY}}
    model: "{model}"
    api_base: "{base_url.rstrip('/')}"
    model_supports_json: true
    temperature: 0
    concurrent_requests: 8
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 3
    request_timeout: {int(global_spec['request_timeout_seconds'])}.0
  default_embedding_model:
    type: embedding
    model_provider: openai
    auth_type: api_key
    api_key: ${{GRAPHRAG_API_KEY}}
    model: unused-no-embedding-model
    api_base: "{base_url.rstrip('/')}"
    concurrent_requests: 1
    async_mode: threaded
    retry_strategy: exponential_backoff
    max_retries: 1
    request_timeout: {int(global_spec['request_timeout_seconds'])}.0

input:
  storage:
    type: file
    base_dir: input
  file_type: json
  file_pattern: "corpus\\\\.json"
  text_column: text
  title_column: title
  metadata: [title]

chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]

output:
  type: file
  base_dir: output

cache:
  type: file
  base_dir: cache

reporting:
  type: file
  base_dir: logs

workflows:
  - load_input_documents
  - create_base_text_units
  - create_final_documents
  - extract_graph
  - finalize_graph
  - extract_covariates
  - create_communities
  - create_final_text_units
  - create_community_reports

vector_store:
  default_vector_store:
    type: lancedb
    db_uri: output/lancedb
    container_name: default
    overwrite: true

embed_text:
  model_id: default_embedding_model
  vector_store_id: default_vector_store
  batch_size: 16
  batch_max_tokens: 8191

extract_graph:
  model_id: default_chat_model
  prompt: extract_graph.prompt
  entity_types: [{entity_types}]
  max_gleanings: 0

summarize_descriptions:
  model_id: default_chat_model
  max_length: 300
  max_input_tokens: 3200

extract_claims:
  enabled: false

community_reports:
  model_id: default_chat_model
  max_length: 1000
  max_input_length: 2800

cluster_graph:
  max_cluster_size: {int(global_spec['max_cluster_size'])}
  use_lcc: false

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: true
  embeddings: false
  raw_graph: false

local_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model

global_search:
  chat_model_id: default_chat_model
  map_prompt: global_map.prompt
  reduce_prompt: global_reduce.prompt

drift_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model

basic_search:
  chat_model_id: default_chat_model
  embedding_model_id: default_embedding_model
'''


def prepare_workspace(
    workspace: Path,
    corpus_path: Path,
    dataset_spec: dict[str, Any],
    global_spec: dict[str, Any],
    model: str,
    base_url: str,
) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    corpus = list(iter_jsonl(corpus_path))
    write_json_atomic(input_dir / "corpus.json", corpus)

    graph = dataset_spec["graphrag"]
    shutil.copyfile(PROMPT_ROOT / graph["extract_prompt"], workspace / "extract_graph.prompt")
    shutil.copyfile(PROMPT_ROOT / graph["map_prompt"], workspace / "global_map.prompt")
    shutil.copyfile(PROMPT_ROOT / graph["reduce_prompt"], workspace / "global_reduce.prompt")
    settings = workspace / "settings.yaml"
    settings.write_text(
        settings_text(dataset_spec, global_spec, model, base_url), encoding="utf-8"
    )
    return settings


def build_index(
    workspace: Path,
    settings: Path,
    api_key: str,
    log_path: Path,
) -> None:
    env = dict(os.environ)
    env["GRAPHRAG_API_KEY"] = api_key
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "graphrag",
        "index",
        "--root",
        str(workspace),
        "--config",
        str(settings),
        "--skip-validation",
        "--no-cache",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PACKAGE_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"GraphRAG indexing failed; inspect {log_path}")


async def load_index(settings: Path, workspace: Path) -> tuple[Any, dict[str, Any]]:
    from graphrag.config.load_config import load_config as load_graphrag_config
    from graphrag.utils.api import create_storage_from_config
    from graphrag.utils.storage import load_table_from_storage

    graph_config = load_graphrag_config(workspace.resolve(), settings.resolve(), {})
    storage = create_storage_from_config(graph_config.output)
    index = {
        name: await load_table_from_storage(name=name, storage=storage)
        for name in ("entities", "communities", "community_reports")
    }
    return graph_config, index


async def generate_outputs(
    workspace: Path,
    settings: Path,
    queries_path: Path,
    output_path: Path,
    dataset: str,
    response_type: str,
    community_level: int,
    concurrency: int,
    per_query_timeout: float,
    limit: int | None,
    resume: bool,
) -> None:
    import graphrag.api as api

    queries = load_queries(queries_path, limit)
    done = completed_ids(output_path) if resume else set()
    pending = [(qid, question) for qid, question in queries if qid not in done]
    if not resume:
        output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pending:
        return

    graph_config, index = await load_index(settings, workspace)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def ask(qid: str, question: str) -> dict[str, Any]:
        async with semaphore:
            try:
                request = api.global_search(
                    config=graph_config,
                    entities=index["entities"],
                    communities=index["communities"],
                    community_reports=index["community_reports"],
                    community_level=community_level,
                    dynamic_community_selection=False,
                    response_type=response_type,
                    query=question,
                    verbose=False,
                )
                if per_query_timeout > 0:
                    response, _ = await asyncio.wait_for(
                        request, timeout=per_query_timeout
                    )
                else:
                    response, _ = await request
                output = response if isinstance(response, str) else json.dumps(response)
                if not output.strip():
                    raise RuntimeError("vLLM returned an empty GraphRAG response")
                return {
                    "id": qid,
                    "_id": qid,
                    "dataset": dataset,
                    "question": question,
                    "output": output,
                    "method": "global",
                }
            except Exception as exc:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                return {
                    "id": qid,
                    "_id": qid,
                    "dataset": dataset,
                    "question": question,
                    "output": "",
                    "method": "global",
                    "timed_out": isinstance(exc, asyncio.TimeoutError),
                    "error": f"{type(exc).__name__}: {exc}",
                }

    tasks = [asyncio.create_task(ask(qid, question)) for qid, question in pending]
    failures = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for task in asyncio.as_completed(tasks):
            record = await task
            failures += int(bool(record.get("error")))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    if failures:
        raise RuntimeError(
            f"{failures} GraphRAG queries failed; rerun this run-id with --resume"
        )


def run_condition(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset_spec = config["datasets"][args.dataset]
    workspace = args.run_dir / "indexes" / "GraphRAG" / args.dataset / args.condition
    if workspace.exists() and not args.resume:
        shutil.rmtree(workspace)
    corpus_path = PACKAGE_ROOT / "data" / args.dataset / args.condition / "corpus.jsonl"
    queries_path = PACKAGE_ROOT / "data" / args.dataset / "queries.jsonl"
    if not corpus_path.is_file() or not queries_path.is_file():
        raise FileNotFoundError("Build the datasets before running GraphRAG")
    settings = prepare_workspace(
        workspace,
        corpus_path,
        dataset_spec,
        config["graphrag"],
        args.model,
        args.base_url,
    )
    index_ready = (workspace / "output" / "community_reports.parquet").is_file()
    log_path = args.run_dir / "logs" / "GraphRAG" / args.dataset / f"{args.condition}.log"
    if not args.resume or not index_ready:
        build_index(workspace, settings, args.api_key, log_path)
    os.environ["GRAPHRAG_API_KEY"] = args.api_key
    asyncio.run(
        generate_outputs(
            workspace=workspace,
            settings=settings,
            queries_path=queries_path,
            output_path=output_file(args.run_dir, "GraphRAG", args.dataset, args.condition),
            dataset=args.dataset,
            response_type=dataset_spec["graphrag"]["response_type"],
            community_level=int(config["graphrag"]["community_level"]),
            concurrency=int(args.query_concurrency),
            per_query_timeout=float(config["graphrag"]["per_query_timeout_seconds"]),
            limit=args.limit,
            resume=args.resume,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GraphRAG main-experiment condition.")
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--condition", required=True, choices=("Clean", "TopoPoison"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--query-concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Set VLLM_API_KEY or pass --api-key")
    run_condition(args)


if __name__ == "__main__":
    main()
