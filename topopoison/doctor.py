from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from .config import PACKAGE_ROOT, load_config
from .vllm_client import VLLMClient


def run_check(python: Path, source: str, label: str) -> str:
    if not python.is_file():
        raise FileNotFoundError(f"{label} Python executable not found: {python}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    result = subprocess.run(
        [str(python), "-c", source],
        cwd=PACKAGE_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} dependency check failed:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def check_environment(
    systems: Iterable[str],
    graph_python: Path,
    hippo_python: Path,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, str]:
    selected = set(systems)
    config = load_config()
    report: dict[str, str] = {}
    client = VLLMClient(base_url, api_key, model, timeout=20)
    client.check()
    vllm_version = client.version()
    expected_vllm = str(config["vllm"]["version"])
    if vllm_version != expected_vllm:
        raise RuntimeError(f"vLLM {vllm_version or 'unknown'} found; expected {expected_vllm}")
    report["vLLM"] = f"{vllm_version}; {model} at {base_url}"

    if "GraphRAG" in selected:
        graph_version = run_check(
            graph_python,
            "import importlib.metadata; print(importlib.metadata.version('graphrag'))",
            "GraphRAG",
        )
        expected_graph = str(config["graphrag"]["version"])
        if graph_version != expected_graph:
            raise RuntimeError(f"GraphRAG {graph_version} found; expected {expected_graph}")
        report["GraphRAG"] = graph_version
    if "HippoRAG2" in selected:
        hippo_versions = {
            "accelerate": str(config["hipporag2"]["accelerate_version"]),
            "torch": str(config["hipporag2"]["torch_version"]),
            "transformers": str(config["hipporag2"]["transformers_version"]),
            "openai": str(config["hipporag2"]["openai_version"]),
        }
        report["HippoRAG2"] = run_check(
            hippo_python,
            "import importlib.metadata as metadata; "
            f"expected={hippo_versions!r}; "
            "found={name: metadata.version(name) for name in expected}; "
            "comparable={name: (value.split('+', 1)[0] if name == 'torch' else value) "
            "for name, value in found.items()}; "
            "assert comparable == expected, f'version mismatch: expected {expected}, found {found}'; "
            "from topopoison.hipporag_runner import install_online_import_stubs, add_vendored_hipporag; "
            "install_online_import_stubs(); add_vendored_hipporag(); "
            "import hipporag, torch, transformers; "
            "assert torch.cuda.is_available(), 'CUDA is required for the HippoRAG2 main experiment'; "
            "free,total=torch.cuda.mem_get_info(); "
            "print('vendored; ' + '; '.join(f'{name}={value}' for name,value in found.items()) + '; gpu=' + torch.cuda.get_device_name(0) + '; free_gib=' + f'{free / 2**30:.1f}')",
            "HippoRAG2",
        )
    return report
