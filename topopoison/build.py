from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_CONFIG, PACKAGE_ROOT, load_config, package_path, select_datasets
from .io import iter_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


def document_id(row: dict[str, Any]) -> str:
    value = row.get("_id") or row.get("id")
    if value is None:
        raise ValueError(f"Document has no id: {row!r}")
    return str(value)


def document_changed(clean: dict[str, Any], candidate: dict[str, Any]) -> bool:
    clean_text = clean.get("text") or clean.get("contents") or ""
    candidate_text = candidate.get("text") or candidate.get("contents") or ""
    return clean_text != candidate_text or clean.get("title") != candidate.get("title")


def build_dataset(
    name: str,
    spec: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    clean_rows = list(iter_jsonl(package_path(spec["clean_corpus"])))
    candidate_rows = list(iter_jsonl(package_path(spec["candidate_corpus"])))
    clean_by_id = {document_id(row): row for row in clean_rows}
    candidate_by_id = {document_id(row): row for row in candidate_rows}

    inserted_ids = sorted(set(candidate_by_id) - set(clean_by_id))
    modified_ids = sorted(
        doc_id
        for doc_id in set(clean_by_id) & set(candidate_by_id)
        if document_changed(clean_by_id[doc_id], candidate_by_id[doc_id])
    )

    budget = int(spec["changed_document_budget"])
    modified_count = int(spec["modified_document_count"])
    inserted_count = budget - modified_count
    target_id = str(spec["target_id"])

    if target_id not in modified_ids:
        raise ValueError(f"{name}: target source document is not available for rewriting")
    if modified_count > len(modified_ids) or inserted_count > len(inserted_ids):
        raise ValueError(f"{name}: candidate pool cannot satisfy the configured budget")

    generator = random.Random(int(spec["sampling_seed"]))
    remaining_modified = [doc_id for doc_id in modified_ids if doc_id != target_id]
    selected_modified = sorted(
        [target_id] + generator.sample(remaining_modified, modified_count - 1)
    )
    selected_inserted = sorted(generator.sample(inserted_ids, inserted_count))
    selected_ids = set(selected_modified) | set(selected_inserted)

    output_rows: list[dict[str, Any]] = []
    for row in clean_rows:
        doc_id = document_id(row)
        output_rows.append(candidate_by_id[doc_id] if doc_id in selected_ids else row)
    output_rows.extend(candidate_by_id[doc_id] for doc_id in selected_inserted)
    output_rows.sort(key=document_id)

    dataset_dir = output_root / name
    clean_dir = dataset_dir / "Clean"
    poison_dir = dataset_dir / "TopoPoison"
    clean_path = clean_dir / "corpus.jsonl"
    corpus_path = poison_dir / "corpus.jsonl"
    query_path = dataset_dir / "queries.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    clean_dir.mkdir(parents=True, exist_ok=True)
    poison_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(package_path(spec["clean_corpus"]), clean_path)
    write_jsonl_atomic(corpus_path, output_rows)
    shutil.copyfile(package_path(spec["queries"]), query_path)

    clean_hash = sha256_file(clean_path)
    corpus_hash = sha256_file(corpus_path)
    query_hash = sha256_file(query_path)
    expected_clean_hash = str(spec["clean_corpus_sha256"])
    expected_hash = str(spec["topopoison_corpus_sha256"])
    expected_query_hash = str(spec["queries_sha256"])
    if clean_hash != expected_clean_hash:
        raise ValueError(
            f"{name}: clean corpus hash mismatch, expected {expected_clean_hash}, received {clean_hash}"
        )
    if corpus_hash != expected_hash:
        raise ValueError(
            f"{name}: corpus hash mismatch, expected {expected_hash}, received {corpus_hash}"
        )
    if query_hash != expected_query_hash:
        raise ValueError(
            f"{name}: query hash mismatch, expected {expected_query_hash}, received {query_hash}"
        )

    manifest = {
        "method": "TopoPoison",
        "dataset": name,
        "target_id": target_id,
        "sampling_seed": int(spec["sampling_seed"]),
        "clean_documents": len(clean_rows),
        "candidate_documents": len(candidate_rows),
        "changed_document_budget": budget,
        "modified_documents": len(selected_modified),
        "inserted_documents": len(selected_inserted),
        "output_documents": len(output_rows),
        "selected_document_ids": sorted(selected_ids),
        "clean_corpus_sha256": clean_hash,
        "topopoison_corpus_sha256": corpus_hash,
        "queries_sha256": query_hash,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def build_datasets(
    names: Iterable[str] | None = None,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    selected = select_datasets(config, names)
    destination = output_root or PACKAGE_ROOT / "data"
    manifests = [
        build_dataset(name, config["datasets"][name], destination) for name in selected
    ]
    write_json_atomic(
        destination / "build_summary.json",
        {"method": "TopoPoison", "datasets": manifests},
    )
    return manifests
