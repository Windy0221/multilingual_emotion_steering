#!/usr/bin/env python3
"""Normalize BRIGHTER and build reproducible per-emotion train samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .inspect import EMOTIONS, LANGUAGES, SPLITS, _is_missing
from emotion_steering.utils.paths import resolve_path


NORMALIZED_COLUMNS = ("id", "lang", "language", "split", "text", *EMOTIONS,
                      "emotion_labels", "label_mask", "source_dataset")
BINARY_COLUMNS = ("id", "text", "lang", "emotion", "label", *EMOTIONS)


def _source_files(source: Path, lang: str, split: str) -> list[Path]:
    """列出数据源目录中的文件。"""
    return sorted((source / lang).glob(f"{split}-*.parquet"))


def _load_source(source: Path, lang: str, split: str) -> pd.DataFrame:
    """读取一个原始数据文件。"""
    files = _source_files(source, lang, split)
    if not files:
        return pd.DataFrame(columns=["id", "text", *EMOTIONS])
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _normalize(frame: pd.DataFrame, lang: str, split: str) -> pd.DataFrame:
    """清理并统一单条情绪记录。"""
    missing = [column for column in ("id", "text", *EMOTIONS) if column not in frame]
    if frame.empty and missing:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    if missing:
        raise ValueError(f"{lang}/{split} is missing required columns: {', '.join(missing)}")
    normalized = frame.loc[:, ["id", "text", *EMOTIONS]].copy()
    normalized["id"] = normalized["id"].astype("string")
    normalized["text"] = normalized["text"].astype("string")
    # Null or illegal values remain nullable; they are never silently changed to 0.
    for emotion in EMOTIONS:
        values = pd.to_numeric(normalized[emotion], errors="coerce")
        invalid = normalized[emotion].map(lambda value: not _is_missing(value)) & ~values.isin([0, 1])
        values[invalid] = pd.NA
        normalized[emotion] = values.astype("Int8")
    usable = (
        normalized["id"].notna() & normalized["id"].str.strip().ne("") &
        normalized["text"].notna() & normalized["text"].str.strip().ne("")
    )
    normalized = normalized.loc[usable].copy()
    normalized.insert(1, "lang", lang)
    normalized.insert(2, "split", split)
    normalized.insert(2, "language", lang)
    normalized["emotion_labels"] = normalized.apply(
        lambda row: [int(row[e]) if not _is_missing(row[e]) else None for e in EMOTIONS], axis=1
    )
    normalized["label_mask"] = normalized.apply(
        lambda row: [not _is_missing(row[e]) for e in EMOTIONS], axis=1
    )
    normalized["source_dataset"] = "BRIGHTER"
    return normalized.loc[:, NORMALIZED_COLUMNS].reset_index(drop=True)


def preprocess(source: Path, output: Path, per_class: int = 50, seed: int = 42,
               sampling_mode: str = "balanced") -> dict[str, Any]:
    """预处理原始数据并保存采样结果。"""
    if per_class < 0:
        raise ValueError("per_class must be non-negative")
    if sampling_mode not in {"balanced", "all_available", "fixed_count"}:
        raise ValueError("sampling_mode must be balanced, all_available, or fixed_count")
    manifest: dict[str, Any] = {
        "source": str(source), "seed": seed, "per_class_target": per_class,
        "sampling_mode": sampling_mode,
        "sampling": "train only; independently sample label 1 and label 0 without replacement",
        "languages": {},
    }
    availability: list[dict[str, Any]] = []

    for lang in LANGUAGES:
        lang_info: dict[str, Any] = {"splits": {}, "emotions": {}}
        for split in SPLITS:
            raw = _load_source(source, lang, split)
            normalized = _normalize(raw, lang, split)
            out_dir = output / "normalized" / lang
            out_dir.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(out_dir / f"{split}.parquet", index=False)
            lang_info["splits"][split] = {
                "source_rows": len(raw), "usable_rows": len(normalized),
                "excluded_rows": len(raw) - len(normalized),
            }

            if split != "train":
                continue
            binary_dir = output / "binary_train" / lang
            binary_dir.mkdir(parents=True, exist_ok=True)
            for emotion in EMOTIONS:
                known = normalized[normalized[emotion].isin([0, 1])]
                chosen_ids: dict[str, list[str]] = {}
                chosen_frames: list[pd.DataFrame] = []
                class_counts: dict[str, dict[str, int]] = {}
                for label in (1, 0):
                    candidates = known[known[emotion] == label]
                    ids = candidates["id"].astype(str).tolist()
                    rng = random.Random(f"{seed}:{lang}:{emotion}:{label}")
                    if sampling_mode == "all_available":
                        sampled_ids = ids.copy()
                    else:
                        class_target = per_class
                        if sampling_mode == "balanced":
                            class_target = min(per_class, int(known[emotion].value_counts().reindex([0, 1], fill_value=0).min()))
                        sampled_ids = rng.sample(ids, min(class_target, len(ids)))
                    selected = candidates.set_index(candidates["id"].astype(str)).loc[sampled_ids].reset_index(drop=True)
                    selected = selected.copy()
                    selected["emotion"] = emotion
                    selected["label"] = label
                    chosen_frames.append(selected.loc[:, BINARY_COLUMNS])
                    chosen_ids[str(label)] = sampled_ids
                    class_counts[str(label)] = {"available": len(ids), "selected": len(sampled_ids)}
                samples = pd.concat(chosen_frames, ignore_index=True) if chosen_frames else pd.DataFrame(columns=BINARY_COLUMNS)
                # Stable ordering, independent of parquet row order after selection.
                samples.to_parquet(binary_dir / f"{emotion}.parquet", index=False)
                lang_info["emotions"][emotion] = {
                    "counts": class_counts,
                    "selected_ids": chosen_ids,
                    "total_selected": len(samples),
                }
                availability.append({
                    "language": lang, "emotion": emotion,
                    "available": class_counts["0"]["available"] > 0 and class_counts["1"]["available"] > 0,
                    "reason": None if class_counts["0"]["available"] > 0 and class_counts["1"]["available"] > 0 else "no_valid_binary_labels",
                    "positive_available": class_counts["1"]["available"],
                    "negative_available": class_counts["0"]["available"],
                    "positive_selected": class_counts["1"]["selected"],
                    "negative_selected": class_counts["0"]["selected"],
                })
        manifest["languages"][lang] = lang_info

    output.mkdir(parents=True, exist_ok=True)
    (output / "sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "concept_availability.json").write_text(
        json.dumps(availability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    """解析命令行参数并运行对应任务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=resolve_path("data/raw/brighter"))
    parser.add_argument("--output-dir", type=Path, default=resolve_path("data/processed/brighter"))
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-mode", choices=("balanced", "all_available", "fixed_count"), default="balanced")
    args = parser.parse_args()
    result = preprocess(args.input_dir, args.output_dir, args.per_class, args.seed, args.sampling_mode)
    print(f"Processed {len(result['languages'])} languages into {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
