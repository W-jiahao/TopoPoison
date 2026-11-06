from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .config import DEFAULT_CONFIG, load_config, package_path, select_datasets
from .io import iter_jsonl, write_json_atomic, write_tsv_atomic


ENTITY_PATTERNS = {
    "business": re.compile(r"\bbusiness\s*[-#:]*\s*(\d+)\b", re.IGNORECASE),
    "item": re.compile(r"\bitem\s*[-#:]*\s*(\d+)\b", re.IGNORECASE),
}
CONDITIONS = ("Clean", "TopoPoison")
SYSTEMS = ("GraphRAG", "HippoRAG2")


def record_id(row: dict[str, Any]) -> str | None:
    value = row.get("_id") or row.get("id")
    return None if value is None else str(value)


def entity_id(value: str | int, entity_type: str) -> str:
    text = str(value).strip()
    lower = text.lower()
    dashed = f"{entity_type}-"
    spaced = f"{entity_type} "
    if lower.startswith(dashed):
        suffix = text.split("-", 1)[1].strip()
    elif lower.startswith(spaced):
        suffix = text.split(None, 1)[1].strip()
    else:
        suffix = text
    if not suffix.isdigit():
        raise ValueError(f"Invalid {entity_type} id: {value!r}")
    return f"{entity_type}-{int(suffix)}"


def parse_ranked_ids(text: str, entity_type: str, k: int = 10) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for match in ENTITY_PATTERNS[entity_type].finditer(text):
        value = entity_id(match.group(1), entity_type)
        if value in seen:
            continue
        ranked.append(value)
        seen.add(value)
        if len(ranked) == k:
            break
    return ranked


def query_ids(path: Path, limit: int | None = None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        qid = record_id(row)
        if qid is not None and qid not in seen:
            values.append(qid)
            seen.add(qid)
        if limit is not None and len(values) >= limit:
            break
    return values


def output_file(run_dir: Path, system: str, dataset: str, condition: str) -> Path:
    return run_dir / "outputs" / system / dataset / f"{condition}.jsonl"


def output_ranks(
    path: Path,
    qids: list[str],
    target_id: str,
    entity_type: str,
    k: int,
) -> tuple[dict[str, int | None], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Generated output not found: {path}")
    allowed = set(qids)
    answers: dict[str, str] = {}
    flags: dict[str, bool] = {}
    for row in iter_jsonl(path):
        qid = record_id(row)
        if qid is None or qid not in allowed:
            continue
        answer = str(row.get("output") or row.get("answer") or row.get("response") or "")
        answers[qid] = answer
        flags[qid] = bool(row.get("timed_out") or row.get("error"))

    ranks: dict[str, int | None] = {}
    for qid in qids:
        identifiers = parse_ranked_ids(answers.get(qid, ""), entity_type, k)
        ranks[qid] = identifiers.index(target_id) + 1 if target_id in identifiers else None
    return ranks, {
        "records": len(answers),
        "missing": len(qids) - len(answers),
        "empty": sum(not answer.strip() for answer in answers.values()),
        "flagged": sum(flags.values()),
    }


def metric_values(ranks: dict[str, int | None], k: int) -> dict[str, float | int]:
    count = len(ranks)
    if count == 0:
        raise ValueError("No queries available for evaluation")
    present = [rank for rank in ranks.values() if rank is not None and rank <= k]
    capped = [rank if rank is not None and rank <= k else k + 1 for rank in ranks.values()]
    return {
        "queries": count,
        "Recall@1": sum(rank == 1 for rank in ranks.values()) / count,
        "Recall@10": len(present) / count,
        "NDCG@10": sum(1.0 / math.log2(rank + 1) for rank in present) / count,
        "MRR@10": sum(1.0 / rank for rank in present) / count,
        "mean_capped_rank": mean(capped),
    }


def paired_values(
    clean: dict[str, int | None],
    poisoned: dict[str, int | None],
    k: int,
) -> dict[str, float | int]:
    qids = [qid for qid in clean if qid in poisoned]
    deltas: list[int] = []
    improved = worsened = unchanged = promoted = 0
    for qid in qids:
        clean_rank = clean[qid]
        poison_rank = poisoned[qid]
        clean_value = clean_rank if clean_rank is not None and clean_rank <= k else k + 1
        poison_value = poison_rank if poison_rank is not None and poison_rank <= k else k + 1
        delta = clean_value - poison_value
        deltas.append(delta)
        improved += int(delta > 0)
        worsened += int(delta < 0)
        unchanged += int(delta == 0)
        clean_hit = clean_rank is not None and clean_rank <= k
        poison_hit = poison_rank is not None and poison_rank <= k
        promoted += int(not clean_hit and poison_hit)
    return {
        "paired_queries": len(qids),
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "promoted_to_top10": promoted,
        "mean_capped_rank_delta": mean(deltas) if deltas else 0.0,
    }


def metric_row(
    dataset: str,
    system: str,
    condition: str,
    metrics: dict[str, float | int],
    diagnostics: dict[str, int],
    paired: dict[str, float | int] | None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "system": system,
        "condition": condition,
        "queries": metrics["queries"],
        "Recall@1": f"{float(metrics['Recall@1']):.6f}",
        "Recall@10": f"{float(metrics['Recall@10']):.6f}",
        "NDCG@10": f"{float(metrics['NDCG@10']):.6f}",
        "MRR@10": f"{float(metrics['MRR@10']):.6f}",
        "mean_capped_rank": f"{float(metrics['mean_capped_rank']):.6f}",
        "records": diagnostics["records"],
        "missing": diagnostics["missing"],
        "empty": diagnostics["empty"],
        "flagged": diagnostics["flagged"],
        "promoted_to_top10": "" if paired is None else paired["promoted_to_top10"],
        "improved": "" if paired is None else paired["improved"],
        "worsened": "" if paired is None else paired["worsened"],
        "unchanged": "" if paired is None else paired["unchanged"],
        "mean_capped_rank_delta": (
            "" if paired is None else f"{float(paired['mean_capped_rank_delta']):.6f}"
        ),
    }


def evaluate_dataset(
    name: str,
    spec: dict[str, Any],
    system: str,
    run_dir: Path,
    k: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    qids = query_ids(package_path(spec["queries"]), limit)
    target_id = entity_id(spec["target_id"], spec["entity_type"])
    clean_ranks, clean_diag = output_ranks(
        output_file(run_dir, system, name, "Clean"),
        qids,
        target_id,
        spec["entity_type"],
        k,
    )
    poison_ranks, poison_diag = output_ranks(
        output_file(run_dir, system, name, "TopoPoison"),
        qids,
        target_id,
        spec["entity_type"],
        k,
    )
    clean_metrics = metric_values(clean_ranks, k)
    poison_metrics = metric_values(poison_ranks, k)
    paired = paired_values(clean_ranks, poison_ranks, k)
    return [
        metric_row(name, system, "Clean", clean_metrics, clean_diag, None),
        metric_row(name, system, "TopoPoison", poison_metrics, poison_diag, paired),
    ]


def evaluate_datasets(
    run_dir: Path,
    names: Iterable[str] | None = None,
    systems: Iterable[str] | None = None,
    limit: int | None = None,
    config_path: Path = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    selected = select_datasets(config, names)
    selected_systems = list(systems) if systems else list(SYSTEMS)
    unknown = [system for system in selected_systems if system not in SYSTEMS]
    if unknown:
        raise ValueError(f"Unknown systems: {', '.join(unknown)}")

    rows: list[dict[str, Any]] = []
    for name in selected:
        for system in selected_systems:
            rows.extend(
                evaluate_dataset(
                    name,
                    config["datasets"][name],
                    system,
                    run_dir,
                    int(config["k"]),
                    limit,
                )
            )

    metric_dir = run_dir / "metrics"
    write_tsv_atomic(metric_dir / "main_metrics.tsv", rows)
    write_json_atomic(
        metric_dir / "main_metrics.json",
        {"method": "TopoPoison", "k": int(config["k"]), "results": rows},
    )
    return rows
