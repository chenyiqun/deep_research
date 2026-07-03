# Qwen + vLLM DeepResearch-Bench Pipeline

This is a compact full-flow implementation for DeepResearch-Bench style experiments:

1. Load DRB tasks and rubrics.
2. Generate research reports with a Qwen model served through vLLM.
3. Use a Qwen model as LLM-as-judge for RACE quality scoring.
4. Optionally run a lightweight FACT citation evaluation.

The implementation is intentionally small and readable, so you can swap prompts, models, or data formats quickly.

## Project Layout

```text
drb_qwen/
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

The async workflow is closer to a real deep-research system:

```text
global information state
-> main agent plans searchable queries
-> web search returns top-k results
-> per-URL reader agents extract core information in parallel
-> query summarizer synthesizes reader notes
-> state updater emits only new/corrected information
-> main agent writes final report from the global state
-> async RACE judge evaluates against DRB reference reports
```

This path uses an OpenAI-compatible vLLM server, so generation and judging can run many requests concurrently without repeatedly loading the model.

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
