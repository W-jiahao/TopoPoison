from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

from .config import (
    DATASET_NAMES,
    DEFAULT_CONFIG,
    PACKAGE_ROOT,
    load_config,
    package_path,
)
from .evaluate import output_file, record_id
from .io import iter_jsonl


def install_online_import_stubs() -> None:
    if "vllm" not in sys.modules:
        module = types.ModuleType("vllm")

        class UnavailableLocalEngine:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("This runner uses the OpenAI-compatible vLLM server")

        class SamplingParams:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        module.LLM = UnavailableLocalEngine
        module.SamplingParams = SamplingParams
        sys.modules["vllm"] = module
    if "outlines" not in sys.modules:
        module = types.ModuleType("outlines")
        module.models = types.ModuleType("outlines.models")
        module.generate = types.ModuleType("outlines.generate")
        sys.modules["outlines"] = module
        sys.modules["outlines.models"] = module.models
        sys.modules["outlines.generate"] = module.generate


def add_vendored_hipporag() -> None:
    source = PACKAGE_ROOT / "third_party" / "HippoRAG" / "src"
    if not (source / "hipporag").is_dir():
        raise FileNotFoundError(f"Vendored HippoRAG source not found: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def load_corpus(path: Path) -> list[str]:
    documents: list[str] = []
    for row in iter_jsonl(path):
        text = str(row.get("text") or row.get("contents") or "").strip()
        if text:
            documents.append(text)
    return documents


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


def recommendation_messages(
    question: str,
    passages: list[str],
    k: int,
    domain: str,
    entity_label: str,
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"Passage {index}:\n{passage}" for index, passage in enumerate(passages, start=1)
    )
    system = (
        f"You are a {domain} recommendation ranking assistant. Use only the retrieved "
        f"evidence and the user query. The query asks for the next {domain} "
        f"{entity_label.lower()}. Return a ranked list of {entity_label.lower()} identifiers, "
        "not prose."
    )
    user = (
        f"Retrieved evidence:\n{context}\n\n"
        f"User query:\n{question}\n\n"
        f"Return exactly {k} unique recommendations in this format:\n"
        f"Rank 1: {entity_label} <id>\n"
        f"Rank 2: {entity_label} <id>\n"
        "...\n"
        f"Rank {k}: {entity_label} <id>\n"
        "Do not include explanations, bullets, citations, or any text outside the ranked lines."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def trim_passages(passages: list[str], max_characters: int) -> list[str]:
    if max_characters <= 0:
        return passages
    return [passage[:max_characters] for passage in passages]


def build_hipporag(
    save_dir: Path,
    dataset: str,
    corpus_len: int,
    base_url: str,
    model: str,
    config: dict[str, Any],
    dataset_spec: dict[str, Any],
    resume: bool,
) -> Any:
    install_online_import_stubs()
    add_vendored_hipporag()
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    common = config["hipporag2"]
    per_dataset = dataset_spec["hipporag2"]
    global_config = BaseConfig(
        save_dir=str(save_dir),
        dataset=dataset,
        llm_name=model,
        llm_base_url=base_url.rstrip("/"),
        embedding_model_name=common["embedding_model"],
        force_index_from_scratch=not resume,
        force_openie_from_scratch=not resume,
        rerank_dspy_file_path=str(package_path(common["rerank_prompt"])),
        retrieval_top_k=int(per_dataset["retrieval_top_k"]),
        linking_top_k=int(common["linking_top_k"]),
        max_qa_steps=1,
        qa_top_k=int(per_dataset["qa_top_k"]),
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=int(common["embedding_batch_size"]),
        embedding_max_seq_len=int(common["embedding_max_sequence_length"]),
        embedding_model_dtype="auto",
        max_new_tokens=int(common["max_generation_tokens"]),
        corpus_len=corpus_len,
        openie_mode="online",
        save_openie=True,
        temperature=0,
    )
    return HippoRAG(global_config=global_config)


def scores_to_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def run_condition(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset_spec = config["datasets"][args.dataset]
    common = config["hipporag2"]
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["HIPPORAG_CONTRIEVER_REVISION"] = str(common["contriever_revision"])

    corpus_path = PACKAGE_ROOT / "data" / args.dataset / args.condition / "corpus.jsonl"
    queries_path = PACKAGE_ROOT / "data" / args.dataset / "queries.jsonl"
    if not corpus_path.is_file() or not queries_path.is_file():
        raise FileNotFoundError("Build the datasets before running HippoRAG2")
    documents = load_corpus(corpus_path)
    queries = load_queries(queries_path, args.limit)
    output_path = output_file(args.run_dir, "HippoRAG2", args.dataset, args.condition)
    done = completed_ids(output_path) if args.resume else set()
    pending = [(qid, question) for qid, question in queries if qid not in done]
    if not args.resume:
        output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pending:
        return

    save_dir = args.run_dir / "indexes" / "HippoRAG2" / args.dataset / args.condition
    hipporag = build_hipporag(
        save_dir=save_dir,
        dataset=args.dataset,
        corpus_len=len(documents),
        base_url=args.base_url,
        model=args.model,
        config=config,
        dataset_spec=dataset_spec,
        resume=args.resume,
    )
    hipporag.index(docs=documents)

    dataset_params = dataset_spec["hipporag2"]
    batch_size = max(1, int(args.query_batch_size))
    failures = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            solutions = hipporag.retrieve(
                queries=[question for _, question in batch],
                num_to_retrieve=int(dataset_params["retrieval_top_k"]),
            )
            for (qid, question), solution in zip(batch, solutions):
                raw_passages = [
                    str(value)
                    for value in solution.docs[: int(dataset_params["qa_top_k"])]
                ]
                passages = trim_passages(
                    raw_passages, int(dataset_params["passage_max_characters"])
                )
                try:
                    output, metadata, cache_hit = hipporag.llm_model.infer(
                        recommendation_messages(
                            question,
                            passages,
                            int(config["k"]),
                            dataset_spec["domain"],
                            dataset_spec["entity_label"],
                        )
                    )
                    output = str(output or "")
                    if not output.strip():
                        raise RuntimeError("vLLM returned an empty HippoRAG2 response")
                    error = None
                except Exception as exc:
                    output = ""
                    metadata = {}
                    cache_hit = False
                    error = f"{type(exc).__name__}: {exc}"
                scores = scores_to_list(solution.doc_scores)
                metadata = dict(metadata or {})
                metadata.update(
                    {
                        "qa_prompt_mode": "recommendation_topk",
                        "recommendation_prompt_variant": "standard",
                        "recommendation_k": int(config["k"]),
                        "cache_hit": bool(cache_hit),
                        "rerank_metadata_dataset": None,
                        "qa_passage_max_chars": int(
                            dataset_params["passage_max_characters"]
                        ),
                        "qa_passages_truncated": sum(
                            len(before) > len(after)
                            for before, after in zip(raw_passages, passages)
                        ),
                        "qa_context_chars": sum(len(passage) for passage in passages),
                        "qa_top_k": int(dataset_params["qa_top_k"]),
                        "qa_passage_count": len(passages),
                    }
                )
                record: dict[str, Any] = {
                    "id": qid,
                    "_id": qid,
                    "dataset": args.dataset,
                    "question": question,
                    "output": output,
                    "raw_response": output,
                    "retrieved": [
                        {
                            "rank": rank,
                            "score": scores[rank - 1] if rank <= len(scores) else None,
                            "text": passage.split("\n", 1)[0],
                        }
                        for rank, passage in enumerate(
                            (str(value) for value in solution.docs), start=1
                        )
                    ],
                    "metadata": metadata,
                }
                if error is not None:
                    record["error"] = error
                    failures += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    if failures:
        raise RuntimeError(
            f"{failures} HippoRAG2 reader queries failed; rerun this run-id with --resume"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one HippoRAG2 main-experiment condition.")
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--condition", required=True, choices=("Clean", "TopoPoison"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--query-batch-size", type=int, default=1)
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
