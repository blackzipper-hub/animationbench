"""Evaluator for Twelve Principles.Squash and Stretch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from animationbench.common.close_set import SYSTEM_PROMPT, dimension_key
from animationbench.common.scoring import compute_scores, is_yes
from animationbench.common.vlm import QwenVideoJudge
from animationbench.models.sam_area_model import analyze_video_area

DIMENSION_KEY = "Twelve Principles.Squash and Stretch"
REBOUND_QUESTION = (
    "Does the video contain a rebound event? Answer only yes or no."
)
DEFAULT_AREA_PROMPT = "ball"
DEFAULT_EPSILON = 1e-6
DEFAULT_DEFORMATION_TAU = 0.2


def average_area_change_rate(
    areas: list[float],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Computes average frame-to-frame relative area change.

    Args:
        areas: Per-frame mask areas.
        epsilon: Small value to avoid division by zero.

    Returns:
        Average relative area-change rate.
    """
    if len(areas) < 2:
        return 0.0
    rates = []
    for previous, current in zip(areas[:-1], areas[1:]):
        denominator = previous + epsilon
        rates.append(abs(current - previous) / denominator)
    return sum(rates) / len(rates) if rates else 0.0


def area_preservation_score(avg_rate: float) -> float:
    """Computes area preservation score S.

    Args:
        avg_rate: Mean per-frame area variation.

    Returns:
        Area preservation score in [0, 100].
    """
    return 100.0 * (1.0 - min(1.0, avg_rate))


def average_deformation_magnitude(anisotropies: list[float]) -> float:
    """Computes mean temporal deformation magnitude.

    Args:
        anisotropies: Per-frame shape anisotropy descriptors.

    Returns:
        Mean absolute descriptor change over time.
    """
    if len(anisotropies) < 2:
        return 0.0
    changes = [
        abs(current - previous)
        for previous, current in zip(anisotropies[:-1], anisotropies[1:])
    ]
    return sum(changes) / len(changes) if changes else 0.0


def deformation_score(
    avg_deformation: float,
    tau: float = DEFAULT_DEFORMATION_TAU,
) -> float:
    """Computes visible deformation reward D.

    Args:
        avg_deformation: Mean temporal deformation magnitude.
        tau: Saturation normalization constant.

    Returns:
        Deformation score in [0, 100].

    Raises:
        ValueError: If tau is not positive.
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")
    return 100.0 * min(1.0, avg_deformation / tau)


def squash_and_stretch_score(
    area_score: float,
    shape_score: float,
) -> float:
    """Combines area preservation and deformation scores into W_2.

    Args:
        area_score: Area preservation score S.
        shape_score: Deformation score D.

    Returns:
        Combined Squash and Stretch score.
    """
    return 0.7 * area_score + 0.3 * shape_score


def evaluate_video(
    video_path: Path,
    judge: QwenVideoJudge,
    output_dir: Path,
    area_prompt: str = DEFAULT_AREA_PROMPT,
    deformation_tau: float = DEFAULT_DEFORMATION_TAU,
) -> dict[str, Any]:
    """Evaluates Squash and Stretch on one video.

    Args:
        video_path: Local video path.
        judge: Qwen video judge for the rebound gate.
        output_dir: Directory for mask/chart artifacts.
        area_prompt: SAM prompt for area tracking.
        deformation_tau: Saturation constant for deformation reward.

    Returns:
        Result payload containing score and visual artifacts.
    """
    rebound_answer = judge.answer_yes_no(
        f"{SYSTEM_PROMPT}\n{REBOUND_QUESTION}",
        video_path,
    )
    if not is_yes(rebound_answer):
        return {
            "answer": f"rebound: {rebound_answer}, score: 0.00",
            "is_correct": None,
            "score": 0.0,
            "avg_rate": 0.0,
            "area_score": 0.0,
            "avg_deformation": 0.0,
            "deformation_score": 0.0,
            "visuals": None,
        }

    model_result = analyze_video_area(
        video_path=video_path,
        prompt_text=area_prompt,
        output_dir=output_dir,
    )
    areas = model_result.get("areas") or []
    anisotropies = model_result.get("anisotropies") or []
    avg_rate = average_area_change_rate(areas)
    avg_deformation = average_deformation_magnitude(anisotropies)
    area_score = area_preservation_score(avg_rate)
    shape_score = deformation_score(avg_deformation, deformation_tau)
    score = squash_and_stretch_score(area_score, shape_score)
    return {
        "answer": (
            f"rebound: {rebound_answer}, score: {score:.2f}, "
            f"area_score: {area_score:.2f}, "
            f"deformation_score: {shape_score:.2f}"
        ),
        "is_correct": None,
        "score": score,
        "avg_rate": avg_rate,
        "area_score": area_score,
        "avg_deformation": avg_deformation,
        "deformation_score": shape_score,
        "anisotropies": anisotropies,
        "visuals": {
            "mask_video": model_result.get("mask_video_path"),
            "area_chart": model_result.get("chart_path"),
        },
    }


def evaluate_questions(
    questions: list[dict[str, Any]],
    video_folder: Path,
    judge: QwenVideoJudge,
    output_dir: Path,
    deformation_tau: float = DEFAULT_DEFORMATION_TAU,
) -> dict[str, Any]:
    """Evaluates Squash and Stretch questions.

    Args:
        questions: Close-set question payloads.
        video_folder: Folder containing referenced videos.
        judge: Qwen video judge.
        output_dir: Directory for mask/chart artifacts.
        deformation_tau: Saturation constant for deformation reward.

    Returns:
        Score and item payload.
    """
    items = []
    for question in questions:
        if dimension_key(question) != DIMENSION_KEY:
            continue
        result = evaluate_video(
            video_path=video_folder / question["video_name"],
            judge=judge,
            output_dir=output_dir,
            deformation_tau=deformation_tau,
        )
        item = {
            "dimension": DIMENSION_KEY,
            "question": question.get("question", ""),
            "video": question["video_name"],
            "answer": result.get("answer", ""),
            "is_correct": result.get("is_correct"),
            "score": result.get("score"),
            "avg_rate": result.get("avg_rate"),
            "area_score": result.get("area_score"),
            "avg_deformation": result.get("avg_deformation"),
            "deformation_score": result.get("deformation_score"),
            "visuals": result.get("visuals"),
        }
        items.append(item)
    return {"scores": compute_scores(items), "items": items}

