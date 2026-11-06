# TopoPoison

TopoPoison is an end-to-end reproduction package for type-plausible topology poisoning in graph retrieval-augmented recommendation. It builds the attack corpus, indexes Clean and TopoPoison data with GraphRAG and HippoRAG2, sends every indexing and recommendation-generation request to API service, and evaluates the outputs produced by that run.

The package covers only the main experiment:

- Datasets: `Amazon`, `Google`, and `Yelp`.
- Conditions: `Clean` and `TopoPoison`.
- Systems: `GraphRAG` and `HippoRAG2`.
- Model: OpenAI-compatible API endpoint.
- Metrics: target Recall@1, Recall@10, NDCG@10, MRR@10, and paired rank movement.

No generated model outputs are shipped in this repository. Every normal reproduction creates a fresh run directory, builds fresh indexes, calls API, and writes the responses returned during that run.

## Repository layout

```text
TopoPoison/
├── README.md
├── LICENSE
├── config.json
├── run.py
├── artifacts/
│   ├── Amazon/
│   ├── Google/
│   └── Yelp/
├── scripts/
│   ├── setup_envs.sh
│   └── start_vllm.sh
├── third_party/
│   └── HippoRAG/
├── topopoison/
│   ├── build.py
│   ├── doctor.py
│   ├── evaluate.py
│   ├── graphrag_runner.py
│   ├── hipporag_runner.py
│   ├── orchestrate.py
│   ├── vllm_client.py
│   └── prompts/graphrag/
└── tests/
```

# Quick Start


## Build the experiment corpora

```bash
python3.10 run.py build
```

This creates:

```text
data/<Dataset>/
├── Clean/corpus.jsonl
├── TopoPoison/corpus.jsonl
├── queries.jsonl
└── manifest.json
```


## Run the full main experiment

```bash
CUDA_VISIBLE_DEVICES=0 \
python3.10 run.py reproduce \
  --base-url http \
  --api-key sk-\
  --model microsoft/phi-4
```

If `--run-id` is omitted, the command creates a timestamped identifier. A full run performs the following work:

Rebuild the three Clean/TopoPoison corpus pairs.
Build six GraphRAG indexes and generate six GraphRAG output files.
Build six HippoRAG2 indexes with online OpenIE and generate six HippoRAG2 output files.
Evaluate the 12 generated output files.

## Resume an interrupted run

Resume is allowed only with an explicit run identifier:

```bash
CUDA_VISIBLE_DEVICES=1 \
python3.10 run.py reproduce \
  --run-id main-seed0 \
  --resume \
  --base-url http \
  --api-key sk-
```

Resume reads indexes and output records from that same run directory. A normal run never reads another run's GraphRAG cache, HippoRAG2 SQLite cache, index, or model output.

Failed or empty query records are not considered complete. The runner stops before evaluation and `--resume` retries only those records while retaining successful work from the same run.

The resume command must keep the original datasets, systems, query limit, model, endpoint, and configuration file. The runner verifies these fields against `manifest.json` before reusing any work.

