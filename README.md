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
tests/
  smoke_test_scoring.py # no-GPU sanity test for scoring logic
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
  --model Qwen/Qwen2.5-7B-Instruct \
  --limit 2 \
  --batch-size 2 \
  --max-tokens 4096
```

For a larger Qwen model, change `--model`, `--tensor-parallel-size`, and `--max-model-len`:

```bash
python -m drb_qwen.generate_reports \
  --query-file data/drb/query.jsonl \
  --output-file outputs/qwen_reports.jsonl \
  --model Qwen/Qwen2.5-72B-Instruct \
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
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --limit 2 \
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
- Use `--resume` for long runs.
- Keep `--temperature 0` for judging.
- Use a stronger Qwen model for judging than for report generation when possible.
- For long reference reports, set a larger `--max-model-len`; otherwise vLLM may truncate or reject prompts.

