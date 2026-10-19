"""Joblib artifacts are executable: load only files you trust."""

import json
import os
from pathlib import Path
from typing import Any

import joblib


def save_artifact(directory: Path, model: Any, metadata: dict, name: str = "url_offline") -> Path:
    """Persist the model and metadata together to avoid mismatched pairs."""
    if name not in {"url_offline", "url_online", "email"}:
        raise ValueError("Unknown model artifact name")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}.joblib"
    temporary = directory / f"{name}.joblib.tmp"
    joblib.dump({"model": model, "metadata": metadata}, temporary)
    os.replace(temporary, destination)
    (directory / f"{name}.metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return destination


def load_artifact(path: Path) -> dict[str, Any]:
    """Load a trusted local artifact with its feature schema."""
    artifact = joblib.load(path)
    if not {"model", "metadata"}.issubset(artifact):
        raise ValueError("Artifact must contain model and metadata")
    return artifact
