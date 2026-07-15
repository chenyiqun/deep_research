# Qwen + vLLM DeepResearch-Bench Pipeline

This is a full-flow implementation for DeepResearch-Bench experiments and arbitrary multi-agent research tasks:

1. Load DRB tasks and rubrics.
2. Generate research reports with a Qwen model served through vLLM.
3. Use a Qwen model as LLM-as-judge for RACE quality scoring.
4. Optionally run a lightweight FACT citation evaluation.

The vLLM, web tooling, multi-agent control plane, state schemas, and evaluation layers are separated so they can be tested or replaced independently.

## Project Layout

```text
drb_qwen/
  multi_agent/          # dynamic DAG, state, agents, tools, audit, persistence
  run_multi_agent_research.py # one arbitrary research task through vLLM serve
  generate_reports_async_research.py # DRB batch multi-agent generation
  generate_reports.py   # task -> report inference pipeline
  evaluate_race.py      # report + reference + rubrics -> RACE score
  evaluate_fact.py      # optional citation extraction / validation
  prompts.py            # report, RACE judge, and FACT prompts
  scoring.py            # weighted score calculation and summaries
  vllm_chat.py          # lazy vLLM chat wrapper
scripts/
  download_drb_data.py  # downloads query/criteria/reference files
  run_qwen3_8b_smoke.sh # server smoke run with Qwen3-8B
  launch_qwen3_8b_smoke_bg.sh # nohup background launcher
tests/
  smoke_test_scoring.py # no-GPU sanity test for scoring logic
```

## Your Server Paths

The scripts are ready for the current server layout:

```text
Repo:
/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research

Data:
/mnt/tidal-alsh01/usr/chenyiqun/datasets/DeepResearch/deep_research_bench_data

Qwen3-8B:
/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-8B
```

Run a no-GPU static check after pulling new code:

```bash
cd /mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research
bash scripts/check_pipeline_static.sh
```

Run a 2-task end-to-end smoke test:

```bash
cd /mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research
bash scripts/run_qwen3_8b_smoke.sh
```

Run it in the background so SSH can disconnect safely:

```bash
cd /mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research
bash scripts/launch_qwen3_8b_smoke_bg.sh
```

The launcher prints the PID and log path. You can monitor with:

```bash
tail -f /mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research/outputs/qwen3_8b_smoke/logs/run_*.log
```

The script writes terminal output to a timestamped log file under:

```text
/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research/outputs/qwen3_8b_smoke/logs/
```

Override the number of tasks if needed:

```bash
LIMIT=5 bash scripts/run_qwen3_8b_smoke.sh
```

Or in the background:

```bash
LIMIT=5 bash scripts/launch_qwen3_8b_smoke_bg.sh
```

Run all 100 tasks on a single 8-GPU node in the background:

```bash
LIMIT=100 \
OUT_DIR=/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research/outputs/qwen3_8b_full100 \
GPU_DEVICES=0,1,2,3,4,5,6,7 \
TENSOR_PARALLEL_SIZE=8 \
bash scripts/launch_qwen3_8b_smoke_bg.sh
```

## Async Multi-Agent Deep Research Workflow

The async path now uses the event-driven architecture described in
[`docs/deep_research_end_to_end_flow.md`](docs/deep_research_end_to_end_flow.md):

```text
ResearchRun + GlobalResearchState
-> Main creates an initial coarse task DAG
-> Scheduler dispatches READY subtasks in parallel
-> each Researcher runs a bounded local ReAct loop
-> every ReAct step is a separate inference request; tools run after that request releases
-> Search / Fetch / Reader produce Source, Evidence, and Claim records
-> deterministic Reducers merge AgentResult objects
-> Main incrementally patches the DAG at strategic boundaries
-> Writer generates only from the evidence packet
-> Citation Auditor passes or creates targeted repair tasks
-> final report remains compatible with RACE and FACT evaluation
```

All Main, Researcher, Reader, Writer, Auditor, and RACE judge inference uses an
OpenAI-compatible vLLM server. Search and URL extraction remain external tools.
The original `deep_research_workflow.py` module is retained as a compatibility import;
the implementation is under `drb_qwen/multi_agent/`.

The runtime persists, per run:

```text
run_state/<run_id>/global_state.json
run_state/<run_id>/events.jsonl
run_state/<run_id>/local/<subtask_id>.json
run_state/<run_id>/checkpoints/<subtask_id>.json
run_state/<run_id>/bundles/<subtask_id>.json
run_state/<run_id>/artifacts/*
```

The checkpoint contains the last fully reduced Researcher step, including its
Source/Evidence/Claim records and usage. A restarted process resumes from that
semantic state; it never depends on chat history or a pinned GPU KV session. If
a terminal Researcher bundle was saved just before a process failure, the
workflow merges that bundle before applying new budget checks.

Reader evidence excerpts must occur in the supplied source text after
whitespace normalization. Invalid/private source URLs and orphaned
Source/Evidence/Claim references are rejected before they enter global state.
Per-wave search allocations enforce the global search-call ceiling even when
multiple Researchers run concurrently.

Start Qwen3-32B as a vLLM server:

```bash
cd /mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research

MODEL_PATH=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-32B \
SERVED_MODEL_NAME=qwen3-32b \
GPU_DEVICES=0,1,2,3,4,5,6,7 \
TENSOR_PARALLEL_SIZE=8 \
MAX_MODEL_LEN=32768 \
bash scripts/start_qwen3_32b_vllm_server.sh
```

The server launcher explicitly enables automatic prefix caching, chunked
prefill, and priority scheduling. Prefix cache is an evictable optimization;
`LocalResearchState` remains the source of truth.

After the server finishes loading, check it:

```bash
curl http://127.0.0.1:8000/v1/models
```

Run the async multi-agent workflow and async RACE judge in the background:

```bash
cd /mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research

export WEB_SEARCH_API_KEY=<your_prod_web_search_key>

LIMIT=100 \
OUT_DIR=/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research/outputs/qwen3_32b_async_research_full100 \
VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
VLLM_MODEL=qwen3-32b \
MAX_CONCURRENT_TASKS=4 \
MAX_CONCURRENT_LLM_CALLS=16 \
URL_FETCH_ENABLED=1 \
URL_VISIT_ENDPOINT= \
MAX_CONCURRENT_URL_FETCHES=16 \
URL_FETCH_TIMEOUT_S=30 \
MAX_ROUNDS=3 \
MAX_SEARCH_QUERIES_PER_ROUND=3 \
SEARCH_TOP_K=5 \
bash scripts/launch_qwen3_32b_async_research_bg.sh
```

Monitor the run:

```bash
tail -f /mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research/outputs/qwen3_32b_async_research_full100/logs/run_*.log
```

Outputs:

```text
qwen3_32b_async_research_reports.jsonl  # report outputs, compatible with evaluate_race
race_raw_results.jsonl                  # async LLM-as-judge raw results
race_summary.json                       # aggregate RACE scores
traces/<id>.json                        # per-task search/read/state trace
run_state/<run_id>/                     # durable global/local state, events, bundles, artifacts
```

### Run one arbitrary research question

```bash
export WEB_SEARCH_API_KEY=<your_prod_web_search_key>

python -m drb_qwen.run_multi_agent_research \
  --prompt "研究企业级 AI Agent 的主要技术路线、市场和风险" \
  --language zh \
  --output-dir outputs/single_research \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --llm-model qwen3-32b \
  --tokenizer-path /mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-32B \
  --url-visit-endpoint http://127.0.0.1:8765/visit
```

The command writes `report.md`, `result.json`, URL cache data, and durable run
state. Add `--resume` to continue an interrupted run.

Important runtime controls include:

- `--max-researchers`: parallel Researcher subtasks inside one run.
- `--max-react-steps`: maximum local ReAct decisions per subtask.
- `--max-subtasks` and `--max-rounds`: dynamic DAG and Main planning limits.
- `--max-total-tool-calls`, `--max-total-searches`, and `--max-total-tokens`: run budgets.
- `--max-audit-rounds`: Citation Audit and targeted repair limit.
- `--run-state-dir` / `--resume-runs`: checkpoint and recovery behavior.
- `--max-model-len`, `--context-safety-tokens`, and `--tokenizer-path`: token-aware input budgeting.
- `--max-concurrent-control-calls`, `--max-concurrent-long-calls`, and `--max-inflight-llm-tokens`: role-aware admission control.
- `--forward-vllm-priority`: forward Main/Researcher/Reader/Writer priority to a vLLM server started with priority scheduling.
- `--disable-structured-outputs`: compatibility escape hatch for older vLLM; the normal path uses JSON Schema.

`URL_FETCH_ENABLED=1` makes the workflow fetch each top-k search result URL and feed the reader the extracted page text. If fetching fails or the extracted text is too short, the reader falls back to the web-search snippet and records the fetch status in `traces/<id>.json`.

If you have an AggAgent-style visit backend, set `URL_VISIT_ENDPOINT=http://host:port` or `URL_VISIT_ENDPOINT=http://host:port/visit`. The workflow will call `POST /visit` with `{"url": ..., "goal": ...}` first, then fall back to direct HTML/PDF fetching unless `URL_VISIT_FALLBACK_ENABLED=0`.

For the no-paid best-effort path, run the bundled visit server with crawl4ai-first HTML extraction, local PDF extraction, and local Qwen/vLLM goal summaries:

```bash
VISIT_ENABLE_CRAWL4AI=1 \
VISIT_HTML_FETCH_MODE=crawl4ai_first \
VISIT_HTML_DIRECT_FALLBACK=0 \
VISIT_CRAWL4AI_TIMEOUT_S=75 \
VISIT_CRAWL4AI_MAX_RETRIES=2 \
VISIT_CRAWL4AI_MAX_CONCURRENCY=1 \
VISIT_SUMMARY_PROVIDER=local_vllm \
VISIT_SUMMARY_BASE_URL=http://127.0.0.1:8000/v1 \
VISIT_SUMMARY_MODEL=qwen3-32b \
VISIT_SUMMARY_MAX_CONCURRENT_REQUESTS=1 \
bash scripts/launch_visit_server_bg.sh
```

This does not call DeepSeek or paid Jina. It uses the same local vLLM server as the report generator to compress each fetched URL into a goal-based visit summary before the reader agent consumes it.

To test URL extraction quality without running the full research workflow:

```bash
WEB_SEARCH_API_KEY=your_key \
PYTHONPATH="$PWD" python scripts/test_search_url_fetch.py \
  --query-file /mnt/tidal-alsh01/usr/chenyiqun/datasets/DeepResearch/deep_research_bench_data/query.jsonl \
  --query-limit 5 \
  --only-lang zh \
  --search-count 15 \
  --search-top-k 8 \
  --url-visit-endpoint http://127.0.0.1:8765/visit \
  --disable-url-visit-fallback \
  --url-visit-timeout-s 150 \
  --max-concurrent-url-fetches 1 \
  --url-fetch-cache-dir outputs/search_url_fetch_test/url_cache \
  --output-file outputs/search_url_fetch_test/url_fetch_results.jsonl \
  --search-results-file outputs/search_url_fetch_test/search_results.jsonl \
  --summary-file outputs/search_url_fetch_test/summary.json \
  --log-file outputs/search_url_fetch_test/logs/run.log
```

Use fewer GPUs by changing both `GPU_DEVICES` and `TENSOR_PARALLEL_SIZE`:

```bash
LIMIT=100 \
OUT_DIR=/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research/outputs/qwen3_8b_full100_4gpu \
GPU_DEVICES=0,1,2,3 \
TENSOR_PARALLEL_SIZE=4 \
bash scripts/launch_qwen3_8b_smoke_bg.sh
```

Use a custom log file if needed:

```bash
LOG_FILE=/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research/outputs/qwen3_8b_smoke/my_run.log \
  bash scripts/run_qwen3_8b_smoke.sh
```

## Install

Use a GPU machine for actual vLLM inference.

```bash
cd drb_qwen_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Core Data

```bash
python scripts/download_drb_data.py --output-dir data/drb
```

This downloads:

- `data/drb/query.jsonl`
- `data/drb/criteria.jsonl`
- `data/drb/reference.jsonl`

## Step 1: Generate Reports With Qwen

Start with a tiny run:

```bash
python -m drb_qwen.generate_reports \
  --query-file data/drb/query.jsonl \
  --output-file outputs/qwen_reports.jsonl \
  --model /mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-8B \
  --limit 2 \
  --batch-size 1 \
  --max-model-len 32768 \
  --max-tokens 8192
```

For a larger Qwen model, change `--model`, `--tensor-parallel-size`, and `--max-model-len`:

```bash
python -m drb_qwen.generate_reports \
  --query-file data/drb/query.jsonl \
  --output-file outputs/qwen_reports.jsonl \
  --model Qwen/Qwen2.5-72B-Instruct \
  --gpu-devices 0,1,2,3 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --resume
```

Output JSONL schema:

```json
{"id": 1, "topic": "...", "language": "zh", "prompt": "...", "article": "...", "model": "..."}
```

## Step 2: RACE LLM-as-Judge Evaluation

This compares each generated report against the DRB reference report using the task-specific rubric.

```bash
python -m drb_qwen.evaluate_race \
  --query-file data/drb/query.jsonl \
  --criteria-file data/drb/criteria.jsonl \
  --target-file outputs/qwen_reports.jsonl \
  --reference-file data/drb/reference.jsonl \
  --output-file outputs/race_raw_results.jsonl \
  --summary-file outputs/race_summary.json \
  --judge-model /mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-8B \
  --limit 2 \
  --max-model-len 32768 \
  --max-tokens 8192
```

Important behavior:

- The judge sees criterion text and explanations, but not weights.
- The judge returns per-criterion scores for target and reference reports.
- `scoring.py` applies hidden rubric weights locally.
- Final RACE scores are normalized as `target / (target + reference)`.
- Summary JSON includes both 0-1 scores and percentage fields.

## Step 3: Optional FACT Citation Evaluation

This lightweight implementation uses Qwen to extract cited claims, optionally fetches each URL, and asks Qwen whether each citation supports the claim.

```bash
python -m drb_qwen.evaluate_fact \
  --query-file data/drb/query.jsonl \
  --reports-file outputs/qwen_reports.jsonl \
  --output-file outputs/fact_raw_results.jsonl \
  --summary-file outputs/fact_summary.json \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --limit 2
```

For a quick no-network check:

```bash
python -m drb_qwen.evaluate_fact \
  --query-file data/drb/query.jsonl \
  --reports-file outputs/qwen_reports.jsonl \
  --output-file outputs/fact_raw_results.jsonl \
  --summary-file outputs/fact_summary.json \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --limit 2 \
  --no-fetch-pages
```

The official DRB FACT pipeline uses Jina Reader and its own extractor/dedup/validate prompts. This file is a faithful minimal clone of the idea, not a byte-for-byte leaderboard reproduction.

## Smoke Test Without vLLM

```bash
PYTHONPATH=. python tests/smoke_test_scoring.py
```

## Practical Notes

- Run generation and judging as separate commands. This avoids keeping two large models in GPU memory.
- Qwen3 thinking mode is disabled by default in this code path for cleaner reports and parseable judge JSON. Pass `--enable-thinking` only if you really want it.
- Use `--resume` for long runs.
- Keep `--temperature 0` for judging.
- Use a stronger Qwen model for judging than for report generation when possible.
- For long reference reports, set a larger `--max-model-len`; otherwise vLLM may truncate or reject prompts.
