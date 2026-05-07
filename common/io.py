"""JSON and filesystem helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    """Creates a directory and its parents when missing.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    """Loads JSON, tolerating trailing commas in copied prompt files.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON payload.

    Raises:
        json.JSONDecodeError: If strict and normalized parsing both fail.
    """
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        normalized = re.sub(r",(\s*[\]}])", r"\1", text)
        return json.loads(normalized)


def save_json(path: Path, data: Any) -> None:
    """Writes JSON with stable formatting.

    Args:
        path: Output JSON path.
        data: JSON-serializable payload.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_text(path: Path) -> str:
    """Reads a UTF-8 text file and strips outer whitespace.

    Args:
        path: Text file path.

    Returns:
        Stripped text content.
    """
    return path.read_text(encoding="utf-8").strip()

