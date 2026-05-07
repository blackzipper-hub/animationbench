"""Shared IP dimension evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from animationbench.common.io import save_json
from animationbench.common.scoring import is_yes
from animationbench.common.vlm import QwenVideoJudge

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def collect_videos(
    video_root: Path,
    model_name: str,
    ip_name: str,
) -> list[Path]:
    """Collects videos for one model and IP name.

    Args:
        video_root: Root containing model folders.
        model_name: Video model folder name.
        ip_name: IP character name.

    Returns:
        Sorted video paths.
    """
    model_root = video_root / model_name
    if not model_root.exists():
        return []
    ip_folder = model_root / ip_name
    if ip_folder.is_dir():
        return sorted(
            path for path in ip_folder.iterdir()
            if path.suffix.lower() in VIDEO_EXTS
        )
    ip_key = ip_name.lower()
    return sorted(
        path
        for path in model_root.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTS
        and (ip_key in path.stem.lower() or ip_key in path.name.lower())
    )


def unique_output_path(output_dir: Path, base_name: str) -> Path:
    """Returns a non-existing output path.

    Args:
        output_dir: Output directory.
        base_name: Desired filename.

    Returns:
        Unique output path.

    Raises:
        RuntimeError: If a unique path cannot be found.
    """
    candidate = output_dir / base_name
    if not candidate.exists():
        return candidate
    for index in range(1, 10000):
        candidate = output_dir / (
            f"{Path(base_name).stem}_{index}{Path(base_name).suffix}"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find unique output path for {base_name}")


def question_key(question: dict[str, Any]) -> str:
    """Builds an IP question dimension key.

    Args:
        question: Question payload.

    Returns:
        Fully qualified question key.
    """
    return f"{question['dimension']}.{question['subdimension']}"


def evaluate_questions(
    questions: list[dict[str, Any]],
    video_path: Path,
    judge: QwenVideoJudge,
) -> list[dict[str, Any]]:
    """Evaluates IP questions for one video.

    Args:
        questions: IP question payloads.
        video_path: Local video path.
        judge: Qwen video judge.

    Returns:
        Detailed answer items.
    """
    items = []
    for question in questions:
        answer = judge.answer_yes_no(question["question"], video_path)
        items.append(
            {
                "dimension": question_key(question),
                "question": question["question"],
                "answer": answer,
                "is_correct": is_yes(answer),
            }
        )
    return items


def save_ip_result(
    result_root: Path,
    model_name: str,
    ip_name: str,
    dimension: str,
    video_path: Path,
    items: list[dict[str, Any]],
) -> Path:
    """Saves an IP dimension result payload.

    Args:
        result_root: Root result directory.
        model_name: Video model name.
        ip_name: IP name.
        dimension: IP dimension name.
        video_path: Evaluated video path.
        items: Evaluation result items.

    Returns:
        Written result path.
    """
    output_dir = result_root / model_name
    output_name = f"{model_name}_{ip_name}_{dimension}_answer.json"
    output_path = unique_output_path(output_dir, output_name)
    save_json(
        output_path,
        {
            "model_name": model_name,
            "ip_name": ip_name,
            "dimension": dimension,
            "video": video_path.name,
            "video_path": str(video_path),
            "items": items,
        },
    )
    return output_path

