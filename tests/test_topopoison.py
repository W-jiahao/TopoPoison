from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from topopoison.build import build_datasets
from topopoison.config import load_config, package_path
from topopoison.evaluate import (
    evaluate_datasets,
    output_file,
    output_ranks,
    parse_ranked_ids,
    query_ids,
)
from topopoison.graphrag_runner import completed_ids as graph_completed_ids
from topopoison.graphrag_runner import settings_text
from topopoison.hipporag_runner import completed_ids as hippo_completed_ids
from topopoison.hipporag_runner import recommendation_messages
from topopoison.io import write_jsonl_atomic
from topopoison.openie_json import extract_list_field
from topopoison.orchestrate import contract_mismatches, redact_api_keys
from topopoison.vllm_client import VLLMClient


class ParserTests(unittest.TestCase):
    def test_deduplicates_and_preserves_first_occurrence(self) -> None:
        answer = "Item 9, Item 9, Item 3, Item 7"
        self.assertEqual(parse_ranked_ids(answer, "item"), ["item-9", "item-3", "item-7"])

    def test_keeps_only_the_first_ten_unique_identifiers(self) -> None:
        answer = " ".join(f"Business {value}" for value in range(1, 13))
        self.assertEqual(
            parse_ranked_ids(answer, "business"),
            [f"business-{value}" for value in range(1, 11)],
        )


class OpenIEParserTests(unittest.TestCase):
    def test_reads_nested_json_and_literal_mappings(self) -> None:
        self.assertEqual(
            extract_list_field('Here\'s JSON: {"triples": [["a", "b", "c"]]}', "triples"),
            [["a", "b", "c"]],
        )
        self.assertEqual(
            extract_list_field("{'named_entities': ['Item 7']}", "named_entities"),
            ["Item 7"],
        )

    def test_rejects_executable_expressions(self) -> None:
        with self.assertRaises(ValueError):
            extract_list_field(
                "{'named_entities': __import__('os').system('false')}",
                "named_entities",
            )


class BuildTests(unittest.TestCase):
    def test_builds_both_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = build_datasets(output_root=root)
            for dataset in ("Amazon", "Google", "Yelp"):
                self.assertTrue((root / dataset / "Clean" / "corpus.jsonl").is_file())
                self.assertTrue((root / dataset / "TopoPoison" / "corpus.jsonl").is_file())
        self.assertEqual([row["dataset"] for row in manifests], ["Amazon", "Google", "Yelp"])
        self.assertTrue(all(row["changed_document_budget"] == 77 for row in manifests))


class GeneratedOutputTests(unittest.TestCase):
    def test_evaluates_current_run_files(self) -> None:
        config = load_config()
        spec = config["datasets"]["Amazon"]
        qids = query_ids(Path(__file__).resolve().parents[1] / spec["queries"], limit=2)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            clean = [
                {"id": qid, "dataset": "Amazon", "output": "Item 1"} for qid in qids
            ]
            poison = [
                {"id": qid, "dataset": "Amazon", "output": "Item 13622, Item 1"}
                for qid in qids
            ]
            write_jsonl_atomic(output_file(run_dir, "GraphRAG", "Amazon", "Clean"), clean)
            write_jsonl_atomic(
                output_file(run_dir, "GraphRAG", "Amazon", "TopoPoison"), poison
            )
            rows = evaluate_datasets(
                run_dir, names=["Amazon"], systems=["GraphRAG"], limit=2
            )
        self.assertEqual(rows[1]["Recall@10"], "1.000000")
        self.assertEqual(rows[1]["promoted_to_top10"], 2)

    def test_resume_retries_failures_and_evaluator_uses_latest_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outputs.jsonl"
            write_jsonl_atomic(
                path,
                [
                    {"id": "q1", "output": "", "error": "timeout"},
                    {"id": "q1", "output": "Item 13622"},
                    {"id": "q2", "output": "", "error": "timeout"},
                ],
            )
            self.assertEqual(graph_completed_ids(path), {"q1"})
            self.assertEqual(hippo_completed_ids(path), {"q1"})
            ranks, diagnostics = output_ranks(path, ["q1", "q2"], "item-13622", "item", 10)
        self.assertEqual(ranks, {"q1": 1, "q2": None})
        self.assertEqual(diagnostics["empty"], 1)
        self.assertEqual(diagnostics["flagged"], 1)


class ManifestTests(unittest.TestCase):
    def test_redacts_api_keys_without_mutating_the_command(self) -> None:
        command = ["run.py", "reproduce", "--api-key", "secret", "--limit", "1"]
        self.assertEqual(
            redact_api_keys(command),
            ["run.py", "reproduce", "--api-key", "<redacted>", "--limit", "1"],
        )
        self.assertEqual(command[3], "secret")
        self.assertEqual(
            redact_api_keys(["run.py", "--api-key=secret"]),
            ["run.py", "--api-key=<redacted>"],
        )

    def test_resume_contract_detects_protocol_changes(self) -> None:
        original = {"model": "microsoft/phi-4", "query_limit": 10}
        self.assertEqual(contract_mismatches(original, dict(original)), [])
        self.assertEqual(
            contract_mismatches(original, {**original, "query_limit": 20}),
            ["query_limit"],
        )


class GraphRAGConfigTests(unittest.TestCase):
    def test_settings_point_to_vllm(self) -> None:
        config = load_config()
        text = settings_text(
            config["datasets"]["Amazon"],
            config["graphrag"],
            "microsoft/phi-4",
            "http://127.0.0.1:9000/v1",
        )
        self.assertIn("http://127.0.0.1:9000/v1", text)
        self.assertIn("temperature: 0", text)
        self.assertIn("use_lcc: false", text)
        self.assertIn('file_pattern: "corpus\\\\.json"', text)


class HippoRAGProtocolTests(unittest.TestCase):
    def test_uses_packaged_reranker_and_standard_reader_prompt(self) -> None:
        config = load_config()
        self.assertTrue(package_path(config["hipporag2"]["rerank_prompt"]).is_file())
        messages = recommendation_messages(
            "Recommend the next item.", ["Item 7 profile"], 10, "Amazon", "Item"
        )
        self.assertIn("recommendation ranking assistant", messages[0]["content"])
        self.assertIn("Return exactly 10 unique recommendations", messages[1]["content"])


class VLLMClientTests(unittest.TestCase):
    def test_models_and_chat(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if self.path == "/version":
                    response = {"version": "0.8.5"}
                else:
                    response = {"data": [{"id": "microsoft/phi-4"}]}
                self.wfile.write(json.dumps(response).encode())

            def do_POST(self) -> None:
                size = int(self.headers["Content-Length"])
                json.loads(self.rfile.read(size))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                response = {
                    "model": "microsoft/phi-4",
                    "choices": [
                        {"message": {"content": "Rank 1: Item 13622"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
                self.wfile.write(json.dumps(response).encode())

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = VLLMClient(
                f"http://127.0.0.1:{server.server_port}/v1",
                "test-key",
                "microsoft/phi-4",
            )
            client.check()
            self.assertEqual(client.version(), "0.8.5")
            output, metadata = client.chat([{"role": "user", "content": "test"}])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertEqual(output, "Rank 1: Item 13622")
        self.assertEqual(metadata["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
