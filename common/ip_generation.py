"""Shared IP video-generation helpers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse

import requests

from animationbench.common.io import read_text

PROMPT_FILENAMES = {
    "appearance": "{ip_name}_apperance.txt",
    "expression": "{ip_name}_action.txt",
    "motion": "{ip_name}_prompt.txt",
}


def build_prompt(ip_dir: Path, ip_name: str, dimension: str) -> str:
    """Builds an IP generation prompt for one dimension.

    Args:
        ip_dir: IP asset directory.
        ip_name: IP character name.
        dimension: One of appearance, expression, or motion.

    Returns:
        Prompt text.

    Raises:
        ValueError: If the dimension is unsupported.
    """
    if dimension not in PROMPT_FILENAMES:
        raise ValueError(f"Unsupported IP dimension: {dimension}")
    prompt_file = ip_dir / "video_prompt" / PROMPT_FILENAMES[dimension].format(
        ip_name=ip_name
    )
    return read_text(prompt_file)


def copy_public_image(
    image_path: Path,
    public_image_root: Path,
    folder_name: str,
) -> Path:
    """Copies an image into a public-serving directory.

    Args:
        image_path: Source image.
        public_image_root: Root public image directory.
        folder_name: Folder under the public root.

    Returns:
        Copied image path.
    """
    target_dir = public_image_root / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / image_path.name
    shutil.copy2(image_path, target)
    return target


def unique_path(path: Path) -> Path:
    """Returns a non-existing path derived from the requested path.

    Args:
        path: Desired path.

    Returns:
        Unique path.

    Raises:
        RuntimeError: If no unique path can be found.
    """
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find unique output path for {path}")


def download_video(url: str, output_dir: Path, fallback_name: str) -> Path:
    """Downloads a generated video.

    Args:
        url: Remote video URL.
        output_dir: Output directory.
        fallback_name: Filename when the URL has no mp4 basename.

    Returns:
        Saved video path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name or fallback_name
    if not filename.lower().endswith(".mp4"):
        filename = fallback_name
    target = unique_path(output_dir / filename)
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with target.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
    return target


def generate_video(
    *,
    service: object,
    prompt: str,
    image_path: Path,
    output_dir: Path,
    public_image_root: Path,
    public_folder: str,
    fallback_name: str,
    video_length: int = 5,
    width: int = 1280,
    height: int = 720,
    seed: int = 123,
    camera_fixed: bool = False,
    poll_interval: int = 10,
    max_polls: int = 60,
) -> Path:
    """Generates one IP video using a Pollo-compatible service object.

    Args:
        service: Object exposing generate_video and poll_task_result.
        prompt: Generation prompt.
        image_path: Source image path.
        output_dir: Output directory.
        public_image_root: Public image copy root.
        public_folder: Public folder name for this request.
        fallback_name: Fallback output mp4 filename.
        video_length: Requested video length.
        width: Requested video width.
        height: Requested video height.
        seed: Generation seed.
        camera_fixed: Whether the camera is fixed.
        poll_interval: Seconds between polls.
        max_polls: Maximum poll count.

    Returns:
        Saved video path.

    Raises:
        RuntimeError: If the remote generation fails.
        TimeoutError: If polling exceeds max_polls.
    """
    if not os.getenv("POLLO_API_KEY"):
        raise ValueError("Missing POLLO_API_KEY.")
    if not os.getenv("PUBLIC_DOMAIN"):
        raise ValueError("Missing PUBLIC_DOMAIN.")

    copied_image = copy_public_image(
        image_path,
        public_image_root,
        public_folder,
    )
    submit_result = service.generate_video(
        prompt=prompt,
        mode="i2v",
        input_image_path=str(copied_image),
        symlink_folder=public_folder,
        video_length=video_length,
        width=width,
        height=height,
        seed=seed,
        camera_fixed=camera_fixed,
    )
    task_id = submit_result["pollo_task_id"]
    for _ in range(max_polls):
        result = service.poll_task_result(task_id)
        status = result.get("status")
        if status == "completed":
            return download_video(
                result["video_url"],
                output_dir,
                fallback_name,
            )
        if status == "failed":
            error_message = result.get("error_message", "Pollo task failed.")
            raise RuntimeError(error_message)
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out polling task {task_id}.")

