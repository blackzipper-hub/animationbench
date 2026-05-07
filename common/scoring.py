"""Scoring helpers for yes/no benchmark outputs."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any


def is_yes(text: str) -> bool:
    """Returns whether a VLM answer should be treated as yes.

    Args:
        text: Raw answer text.

    Returns:
        True when the first yes/no token is yes.
    """
    normalized = (text or "").strip().lower()
    if normalized == "yes":
        return True
    match = re.findall(r"\b(yes|no)\b", normalized)
    return bool(match and match[0] == "yes")


def compute_scores(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes per-dimension and overall scores.

    Args:
        items: Evaluation result items.

    Returns:
        Score dictionary with per-dimension and overall values.
    """
    by_dimension = defaultdict(list)
    for item in items:
        by_dimension[item["dimension"]].append(item)

    dimension_scores = {}
    total_score = 0.0
    total_count = 0
    for dimension, dimension_items in by_dimension.items():
        scores = []
        for item in dimension_items:
            if item.get("score") is not None:
                scores.append(float(item["score"]))
            elif item.get("is_correct") is not None:
                scores.append(100.0 if item["is_correct"] else 0.0)
        total_count += len(scores)
        total_score += sum(scores)
        dimension_scores[dimension] = (
            0.0 if not scores else round(sum(scores) / len(scores), 2)
        )

    overall = 0.0 if total_count == 0 else round(total_score / total_count, 2)
    return {"dimensions": dimension_scores, "overall": overall}

