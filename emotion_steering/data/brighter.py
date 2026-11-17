"""Load preprocessed BRIGHTER binary tasks in PsySET's multitask format."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from emotion_steering.utils.paths import resolve_path

logger = logging.getLogger(__name__)

EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
LANGUAGES = ("eng", "esp", "mar", "tat", "hin", "ron", "rus", "hau", "deu", "chn")


def load_brighter_local(
    language: str = "eng",
    split: str = "train",
    emotions: Sequence[str] | None = None,
    data_root: str | Path | None = None,
):
    """Return ``data, labels, concepts, dataset_type`` for PsySET."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported BRIGHTER language {language!r}; choose from {LANGUAGES}")
    if split != "train":
        raise ValueError("BRIGHTER steering samples are restricted to split='train'")
    concepts = list(EMOTIONS if emotions is None else emotions)
    if not concepts or len(set(concepts)) != len(concepts):
        raise ValueError("emotions must contain at least one unique emotion")
    unknown = [emotion for emotion in concepts if emotion not in EMOTIONS]
    if unknown:
        raise ValueError(f"Unsupported emotions: {unknown}")
    root = resolve_path(data_root or "data/processed/brighter")

    data: list[list[dict[str, str]]] = []
    labels: list[np.ndarray] = []
    loaded_concepts: list[str] = []
    for emotion in concepts:
        path = root / "binary_train" / language / f"{emotion}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessed BRIGHTER task not found: {path}. Run `python run.py prepare-data` first.")
        frame = pd.read_parquet(path)
        required = {"text", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["label"].isna().any() or not set(frame["label"].astype(int).unique()).issubset({0, 1}):
            raise ValueError(f"{path} contains invalid binary labels")
        counts = frame["label"].astype(int).value_counts().to_dict()
        if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
            reason = "no valid samples" if frame.empty else "missing positive or negative class"
            logger.warning(
                "Skipping BRIGHTER task language=%s emotion=%s: %s (label counts: 0=%d, 1=%d)",
                language, emotion, reason, counts.get(0, 0), counts.get(1, 0),
            )
            continue
        task_data = [
            {"system": "", "user": str(text), "assistant": ""}
            for text in frame["text"].tolist()
        ]
        data.append(task_data)
        labels.append(frame["label"].astype("int64").to_numpy().reshape(-1, 1))
        loaded_concepts.append(emotion)
    if not loaded_concepts:
        raise ValueError(f"No BRIGHTER tasks for language={language!r} have both binary classes")
    return data, labels, loaded_concepts, "multitask"
