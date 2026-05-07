"""IP motion generation and evaluation."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from animationbench.common import ip_generation
from animationbench.common.ip_evaluation import evaluate_questions
from animationbench.common.vlm import QwenVideoJudge

DIMENSION = "motion"
SUBDIMENSIONS = (
    "Environmental Interaction",
    "Signature",
    "Action Logic",
)


def build_prompt(ip_dir: Path, ip_name: str) -> str:
    """Builds the motion generation prompt.

    Args:
        ip_dir: IP asset directory.
        ip_name: IP character name.

    Returns:
        Prompt text for motion video generation.
    """
    return ip_generation.build_prompt(ip_dir, ip_name, DIMENSION)


def extract_questions(ip_profile: dict[str, Any]) -> list[dict[str, str]]:
    """Extracts motion evaluation questions from an IP profile.

    Args:
        ip_profile: Parsed IP profile payload.

    Returns:
        Motion question payloads.
    """
    behavior = ip_profile.get("canonical_behavior", {})
    return [
        {"dimension": DIMENSION, "subdimension": key, "question": question}
        for key in SUBDIMENSIONS
        for question in behavior.get(key, [])
    ]


def evaluate_video(
    ip_profile: dict[str, Any],
    video_path: Path,
    judge: QwenVideoJudge,
) -> list[dict[str, Any]]:
    """Evaluates motion questions on one video.

    Args:
        ip_profile: Parsed IP profile payload.
        video_path: Local video path.
        judge: Qwen video judge.

    Returns:
        Detailed answer items.
    """
    return evaluate_questions(extract_questions(ip_profile), video_path, judge)


def generate_video(
    *,
    service: object,
    ip_dir: Path,
    ip_name: str,
    image_path: Path,
    output_dir: Path,
    public_image_root: Path,
) -> Path:
    """Generates one motion video for an IP.

    Args:
        service: Pollo-compatible service object.
        ip_dir: IP asset directory.
        ip_name: IP character name.
        image_path: Reference image path.
        output_dir: Output directory.
        public_image_root: Public image copy root.

    Returns:
        Saved generated video path.
    """
    public_folder = f"animationbench_{ip_name}_{DIMENSION}_{int(time.time())}"
    return ip_generation.generate_video(
        service=service,
        prompt=build_prompt(ip_dir, ip_name),
        image_path=image_path,
        output_dir=output_dir,
        public_image_root=public_image_root,
        public_folder=public_folder,
        fallback_name=f"{ip_name}_{DIMENSION}.mp4",
    )

