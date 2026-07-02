from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


FILES = {
    "query.jsonl": "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main/data/prompt_data/query.jsonl",
    "criteria.jsonl": "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main/data/criteria_data/criteria.jsonl",
    "reference.jsonl": "https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/main/data/test_data/cleaned_data/reference.jsonl",
}


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {output}")
    urllib.request.urlretrieve(url, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DeepResearch-Bench core data files.")
    parser.add_argument("--output-dir", default="data/drb")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for name, url in FILES.items():
        download(url, output_dir / name)


if __name__ == "__main__":
    main()

