"""SAM-backed area-change analysis for Squash and Stretch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from animationbench.common.io import ensure_dir


class SamAreaAnalyzer:
    """Analyzes object mask area changes across video frames."""

    def __init__(self, debug: bool = False) -> None:
        """Initializes the analyzer.

        Args:
            debug: Whether to print per-frame debug information.
        """
        self._debug = debug

    def analyze(
        self,
        video_path: Path,
        prompt_text: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Runs SAM area analysis on a video.

        Args:
            video_path: Local video path.
            prompt_text: Text prompt for SAM.
            output_dir: Directory for visual artifacts.

        Returns:
            Area list, FPS, and visual artifact paths.
        """
        ensure_dir(output_dir)
        outputs_per_frame, fps = _run_image_model(
            video_path=video_path,
            prompt_text=prompt_text,
            debug=self._debug,
        )
        mask_video_path = output_dir / f"{video_path.stem}_mask.mp4"
        _create_mask_video(video_path, outputs_per_frame, mask_video_path)
        frame_indices, areas = _calculate_total_area_per_frame(
            outputs_per_frame
        )
        anisotropies = _calculate_shape_anisotropies(outputs_per_frame)
        chart_path = output_dir / f"{video_path.stem}_area_chart.png"
        _draw_area_chart(frame_indices, areas, chart_path, video_fps=fps or 30)
        return {
            "areas": areas,
            "anisotropies": anisotropies,
            "fps": fps,
            "mask_video_path": str(mask_video_path),
            "chart_path": str(chart_path),
        }


def analyze_video_area(
    video_path: Path,
    prompt_text: str,
    output_dir: Path,
    debug: bool = False,
) -> dict[str, Any]:
    """Analyzes total prompted-object area across video frames.

    Args:
        video_path: Local video path.
        prompt_text: Text prompt for SAM.
        output_dir: Directory for visual artifacts.
        debug: Whether to print debug information.

    Returns:
        Area-analysis result payload.
    """
    analyzer = SamAreaAnalyzer(debug=debug)
    return analyzer.analyze(
        video_path=video_path,
        prompt_text=prompt_text,
        output_dir=output_dir,
    )


def _calculate_mask_area(mask: Any) -> int:
    import numpy as np
    import torch

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return int(np.sum(mask > 0))


def _run_image_model(
    video_path: Path,
    prompt_text: str,
    debug: bool = False,
) -> tuple[dict[int, dict[int, Any]], float]:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outputs_per_frame = {}
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt_text,
        )
        frame_outputs = _extract_frame_outputs(
            masks=output.get("masks") or [],
            width=width,
            height=height,
        )
        outputs_per_frame[frame_idx] = frame_outputs
        if debug and frame_idx < 3:
            areas = {
                obj_id: _calculate_mask_area(mask)
                for obj_id, mask in frame_outputs.items()
            }
            print(
                "[DEBUG] frame "
                f"{frame_idx}: obj_count={len(frame_outputs)}, areas={areas}"
            )
        frame_idx += 1

    cap.release()
    return outputs_per_frame, fps


def _extract_frame_outputs(
    masks: Iterable[Any],
    width: int,
    height: int,
) -> dict[int, Any]:
    import cv2
    import numpy as np
    import torch

    frame_outputs = {}
    for obj_id, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array.squeeze()
        mask_bool = mask_array > 0
        if mask_bool.shape != (height, width):
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if mask_bool.sum() > 0:
            frame_outputs[obj_id] = mask_bool
    return frame_outputs


def _calculate_total_area_per_frame(
    outputs_per_frame: dict[int, dict[int, Any]],
) -> tuple[list[int], list[int]]:
    areas = []
    frame_indices = sorted(outputs_per_frame.keys())
    for frame_idx in frame_indices:
        total_area = 0
        for mask in outputs_per_frame[frame_idx].values():
            total_area += _calculate_mask_area(mask)
        areas.append(total_area)
    return frame_indices, areas


def _calculate_shape_anisotropies(
    outputs_per_frame: dict[int, dict[int, Any]],
    epsilon: float = 1e-6,
) -> list[float]:
    import numpy as np

    anisotropies = []
    for frame_idx in sorted(outputs_per_frame.keys()):
        frame_mask = _combine_frame_masks(outputs_per_frame[frame_idx])
        if frame_mask is None:
            anisotropies.append(0.0)
            continue
        coords = np.argwhere(frame_mask > 0)
        if coords.shape[0] < 2:
            anisotropies.append(0.0)
            continue
        covariance = np.cov(coords.astype(float), rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        lambda_1 = float(eigenvalues[-1])
        lambda_2 = float(eigenvalues[0])
        anisotropy = np.log(
            (np.sqrt(lambda_1) + epsilon)
            / (np.sqrt(lambda_2) + epsilon)
        )
        anisotropies.append(float(anisotropy))
    return anisotropies


def _combine_frame_masks(frame_outputs: dict[int, Any]) -> Any:
    import numpy as np
    import torch

    combined_mask = None
    for mask in frame_outputs.values():
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask_bool = np.asarray(mask) > 0
        if combined_mask is None:
            combined_mask = mask_bool.copy()
        else:
            combined_mask = np.logical_or(combined_mask, mask_bool)
    return combined_mask


def _create_mask_video(
    video_path: Path,
    outputs_per_frame: dict[int, dict[int, Any]],
    output_path: Path,
) -> None:
    import cv2
    import torch

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in outputs_per_frame:
            overlay = frame.copy()
            for obj_id, mask in outputs_per_frame[frame_idx].items():
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                overlay[mask > 0] = _color_for_id(obj_id)
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def _color_for_id(obj_id: int) -> tuple[int, int, int]:
    import numpy as np

    np.random.seed(obj_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


def _draw_area_chart(
    frame_indices: Iterable[int],
    areas: Iterable[int],
    output_path: Path,
    video_fps: float = 30,
) -> None:
    import cv2
    import numpy as np

    frame_indices = list(frame_indices)
    areas = list(areas)
    width, height, padding = 1200, 800, 100
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    times = [idx / video_fps for idx in frame_indices] if frame_indices else [0]
    max_time = max(times) if times else 1
    max_area = max(areas) if areas else 1
    min_area = min(areas) if areas else 0

    cv2.line(
        image,
        (padding, height - padding),
        (width - padding, height - padding),
        (0, 0, 0),
        2,
    )
    cv2.line(
        image,
        (padding, padding),
        (padding, height - padding),
        (0, 0, 0),
        2,
    )
    if len(times) > 1:
        points = []
        for time_value, area in zip(times, areas):
            x = int(padding + (time_value / max_time) * (width - 2 * padding))
            y = int(
                height
                - padding
                - ((area - min_area) / (max_area - min_area + 1e-6))
                * (height - 2 * padding)
            )
            points.append([x, y])
        point_array = np.array(points, dtype=np.int32)
        cv2.polylines(image, [point_array], False, (255, 0, 0), 2, cv2.LINE_AA)
        for point in point_array:
            cv2.circle(image, tuple(point), 3, (0, 0, 255), -1)
    cv2.imwrite(str(output_path), image)

