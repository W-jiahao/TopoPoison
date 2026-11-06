#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from topopoison.build import build_datasets
from topopoison.config import DATASET_NAMES, PACKAGE_ROOT, load_config
from topopoison.doctor import check_environment
from topopoison.evaluate import SYSTEMS, evaluate_datasets
from topopoison.orchestrate import reproduce


DEFAULT_GRAPH_PYTHON = PACKAGE_ROOT / ".venvs" / "graphrag" / "bin" / "python"
DEFAULT_HIPPO_PYTHON = PACKAGE_ROOT / ".venvs" / "hipporag" / "bin" / "python"


def add_service_options(parser: argparse.ArgumentParser) -> None:
    config = load_config()
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "sk-vllm"))
    parser.add_argument("--model", default=config["default_model"])
    parser.add_argument("--graphrag-python", type=Path, default=DEFAULT_GRAPH_PYTHON)
    parser.add_argument("--hipporag-python", type=Path, default=DEFAULT_HIPPO_PYTHON)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)


def print_metrics(rows: list[dict[str, object]]) -> None:
    print("Dataset\tSystem\tCondition\tRecall@1\tRecall@10\tNDCG@10\tMRR@10")
    for row in rows:
        print(
            "\t".join(
                str(row[key])
                for key in (
                    "dataset",
                    "system",
                    "condition",
                    "Recall@1",
                    "Recall@10",
                    "NDCG@10",
                    "MRR@10",
                )
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="TopoPoison end-to-end reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build Clean and TopoPoison corpora")
    build_parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES)

    doctor_parser = subparsers.add_parser("doctor", help="Check environments and vLLM")
    add_service_options(doctor_parser)

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Index, generate, and evaluate a fresh run"
    )
    add_service_options(reproduce_parser)
    reproduce_parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES)
    reproduce_parser.add_argument("--run-id")
    reproduce_parser.add_argument("--limit", type=int)
    reproduce_parser.add_argument("--resume", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate generated outputs")
    evaluate_parser.add_argument("--run-id", required=True)
    evaluate_parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES)
    evaluate_parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    evaluate_parser.add_argument("--limit", type=int)

    args = parser.parse_args()
    if args.command == "build":
        manifests = build_datasets(args.datasets)
        for manifest in manifests:
            print(
                f"{manifest['dataset']}: Clean={manifest['clean_documents']}, "
                f"TopoPoison={manifest['output_documents']}, "
                f"changed={manifest['changed_document_budget']}"
            )
    elif args.command == "doctor":
        report = check_environment(
            args.systems or SYSTEMS,
            args.graphrag_python,
            args.hipporag_python,
            args.base_url,
            args.api_key,
            args.model,
        )
        for name, value in report.items():
            print(f"{name}: {value}")
    elif args.command == "reproduce":
        run_dir, rows = reproduce(
            names=args.datasets,
            systems=args.systems,
            run_id=args.run_id,
            graph_python=args.graphrag_python,
            hippo_python=args.hipporag_python,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            limit=args.limit,
            resume=args.resume,
        )
        print(f"Run directory: {run_dir}")
        print_metrics(rows)
    else:
        rows = evaluate_datasets(
            run_dir=PACKAGE_ROOT / "runs" / args.run_id,
            names=args.datasets,
            systems=args.systems,
            limit=args.limit,
        )
        print_metrics(rows)


if __name__ == "__main__":
    main()
