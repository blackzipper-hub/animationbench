"""Shared close-set dimension evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from animationbench.common.scoring import compute_scores, is_yes
from animationbench.common.vlm import QwenVideoJudge

SYSTEM_PROMPT = "answer the following questions, only output yes or no"


def dimension_key(question: dict[str, Any]) -> str:
    """Builds a close-set dimension key.

    Args:
        question: Question payload with dimension and subdimension.

    Returns:
        Fully qualified dimension key.
    """
    return f"{question['dimension']}.{question['subdimension']}"


def evaluate_vlm_questions(
    questions: list[dict[str, Any]],
    video_folder: Path,
    judge: QwenVideoJudge,
    expected_dimension: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Evaluates VLM-only close-set questions for one dimension.

    Args:
        questions: Close-set question payloads.
        video_folder: Folder containing referenced videos.
        judge: Qwen video judge.
        expected_dimension: Dimension key this module owns.
        system_prompt: Prefix prompt for yes/no answers.

    Returns:
        Scores and detailed result items.
    """
    items = []
    for question in questions:
        key = dimension_key(question)
        if key != expected_dimension:
            continue
        video_path = video_folder / question["video_name"]
        prompt = f"{system_prompt}\n{question.get('question', '')}"
        answer = judge.answer_yes_no(prompt, video_path)
        items.append(
            {
                "dimension": key,
                "question": question.get("question", ""),
                "video": question["video_name"],
                "answer": answer,
                "is_correct": is_yes(answer),
            }
        )
    return {"scores": compute_scores(items), "items": items}


def evaluate_single_video_question(
    question: str,
    video_path: Path,
    judge: QwenVideoJudge,
    dimension: str,
) -> dict[str, Any]:
    """Evaluates one ad hoc close-set VLM question.

    Args:
        question: Question text.
        video_path: Local video path.
        judge: Qwen video judge.
        dimension: Dimension key for the result.

    Returns:
        Single-result payload with score fields.
    """
    answer = judge.answer_yes_no(f"{SYSTEM_PROMPT}\n{question}", video_path)
    item = {
        "dimension": dimension,
        "question": question,
        "video": video_path.name,
        "answer": answer,
        "is_correct": is_yes(answer),
    }
    return {"scores": compute_scores([item]), "items": [item]}

