from __future__ import annotations

from drb_qwen.scoring import calculate_weighted_scores, normalize_pair_scores, summarize_race


def main() -> None:
    criteria = {
        "dimension_weight": {
            "comprehensiveness": 0.4,
            "insight": 0.3,
            "instruction_following": 0.2,
            "readability": 0.1,
        },
        "criterions": {
            "comprehensiveness": [
                {"criterion": "coverage", "weight": 0.7},
                {"criterion": "evidence", "weight": 0.3},
            ],
            "insight": [{"criterion": "depth", "weight": 1.0}],
            "instruction_following": [{"criterion": "task fit", "weight": 1.0}],
            "readability": [{"criterion": "clarity", "weight": 1.0}],
        },
    }
    judge_output = {
        "comprehensiveness": [
            {"criterion": "coverage", "article_1_score": 8, "article_2_score": 6},
            {"criterion": "evidence", "article_1_score": 6, "article_2_score": 6},
        ],
        "insight": [{"criterion": "depth", "article_1_score": 7, "article_2_score": 5}],
        "instruction_following": [
            {"criterion": "task fit", "article_1_score": 9, "article_2_score": 7}
        ],
        "readability": [{"criterion": "clarity", "article_1_score": 8, "article_2_score": 8}],
    }
    weighted = calculate_weighted_scores(judge_output, criteria)
    normalized = normalize_pair_scores(weighted)
    summary = summarize_race([{**normalized}])

    assert 0.0 <= normalized["overall_score"] <= 1.0
    assert normalized["overall_score"] > 0.5
    assert summary["n"] == 1.0
    print("smoke_test_scoring passed")
    print(summary)


if __name__ == "__main__":
    main()

