"""Evaluator for Follow Through and Overlapping Action."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from animationbench.common.close_set import (
    evaluate_single_video_question,
    evaluate_vlm_questions,
)
from animationbench.common.vlm import QwenVideoJudge

DIMENSION_KEY = "Twelve Principles.Follow through and overlapping action"


def evaluate_questions(
    questions: list[dict[str, Any]],
    video_folder: Path,
    judge: QwenVideoJudge,
) -> dict[str, Any]:
    """Evaluates Follow Through and Overlapping Action questions.

    Args:
        questions: Close-set question payloads.
        video_folder: Folder containing referenced videos.
        judge: Qwen video judge.

    Returns:
        Score and item payload.
    """
    return evaluate_vlm_questions(
        questions=questions,
        video_folder=video_folder,
        judge=judge,
        expected_dimension=DIMENSION_KEY,
    )


def evaluate_video(
    question: str,
    video_path: Path,
    judge: QwenVideoJudge,
) -> dict[str, Any]:
    """Evaluates one Follow Through question on one video.

    Args:
        question: Question text.
        video_path: Local video path.
        judge: Qwen video judge.

    Returns:
        Single-video result payload.
    """
    return evaluate_single_video_question(
        question=question,
        video_path=video_path,
        judge=judge,
        dimension=DIMENSION_KEY,
    )

