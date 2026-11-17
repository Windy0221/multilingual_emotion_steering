#!/usr/bin/env python3
"""Read-only audit of the BRIGHTER Parquet files used by PsySET."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from emotion_steering.utils.paths import resolve_path


LANGUAGES = ("eng", "esp", "mar", "tat", "hin", "ron", "rus", "hau", "deu", "chn")
SPLITS = ("train", "dev", "test")
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
REQUIRED_COLUMNS = ("id", "text", *EMOTIONS, "emotions")
VALID_LABELS = {0, 1}


def _is_missing(value: Any) -> bool:
    """判断数据值是否为空或缺失。"""
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _emotion_set(value: Any) -> set[str] | None:
    """把情绪标签统一转换为集合。"""
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return {part.strip().lower() for part in value.split(",") if part.strip()}
    try:
        return {str(part).strip().lower() for part in value if str(part).strip()}
    except TypeError:
        return {str(value).strip().lower()}


def _read_split(language_dir: Path, split: str) -> tuple[pd.DataFrame | None, list[Path]]:
    """读取一个数据划分文件。"""
    files = sorted(language_dir.glob(f"{split}-*.parquet"))
    if not files:
        return None, []
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True), files


def inspect(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """检查原始数据并生成质量报告。"""
    split_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    seen_texts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for lang in LANGUAGES:
        lang_dir = input_dir / lang
        for split in SPLITS:
            frame, files = _read_split(lang_dir, split) if lang_dir.is_dir() else (None, [])
            if frame is None:
                split_rows.append({
                    "lang": lang, "split": split, "status": "missing",
                    "files": 0, "rows": 0, "missing_columns": ";".join(REQUIRED_COLUMNS),
                    "empty_text": 0, "missing_id": 0, "duplicate_id_rows": 0,
                    "unique_texts": 0, "duplicate_text_rows": 0,
                    "duplicate_text_extra_rows": 0, "invalid_label_values": 0,
                    "cross_split_duplicate_text_rows": 0,
                    "emotions_mismatch_rows": 0,
                })
                continue

            missing_columns = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
            id_values = frame["id"] if "id" in frame else pd.Series([None] * len(frame))
            text_values = frame["text"] if "text" in frame else pd.Series([None] * len(frame))
            empty_text = sum(_is_missing(v) or not str(v).strip() for v in text_values)
            missing_id = sum(_is_missing(v) or not str(v).strip() for v in id_values)
            duplicate_id_rows = int(id_values.notna().sum() - id_values.dropna().nunique()) if "id" in frame else 0
            normalized_texts = text_values.map(lambda v: str(v).strip() if not _is_missing(v) else "")
            nonempty_texts = normalized_texts[normalized_texts.ne("")]
            text_counts = nonempty_texts.value_counts()
            duplicate_text_rows = int(text_counts[text_counts > 1].sum())
            duplicate_text_extra_rows = int((text_counts[text_counts > 1] - 1).sum())
            cross_split_duplicate_text_rows = 0
            for text in normalized_texts:
                if text:
                    seen_texts[lang][text].add(split)

            invalid_label_values = 0
            emotions_mismatch_rows = 0
            for emotion in EMOTIONS:
                if emotion not in frame:
                    positive = negative = unknown = invalid = 0
                else:
                    col = frame[emotion]
                    missing_mask = col.map(_is_missing)
                    numeric = pd.to_numeric(col, errors="coerce")
                    valid_mask = numeric.isin(VALID_LABELS)
                    invalid = int((~missing_mask & ~valid_mask).sum())
                    positive = int((numeric == 1).sum())
                    negative = int((numeric == 0).sum())
                    unknown = int(missing_mask.sum())
                    invalid_label_values += invalid
                label_rows.append({
                    "lang": lang, "split": split, "emotion": emotion,
                    "positive": positive, "negative": negative, "unknown": unknown,
                    "invalid": invalid,
                })

            if "emotions" in frame and all(emotion in frame for emotion in EMOTIONS):
                for _, row in frame.iterrows():
                    listed = _emotion_set(row["emotions"])
                    if listed is None:
                        continue
                    expected = {e for e in EMOTIONS if not _is_missing(row[e]) and row[e] == 1}
                    known = {e for e in EMOTIONS if not _is_missing(row[e])}
                    if (listed & known) != expected:
                        emotions_mismatch_rows += 1

            split_rows.append({
                "lang": lang, "split": split, "status": "ok",
                "files": len(files), "rows": len(frame),
                "missing_columns": ";".join(missing_columns),
                "empty_text": empty_text, "missing_id": missing_id,
                "duplicate_id_rows": duplicate_id_rows,
                "unique_texts": int(nonempty_texts.nunique()),
                "duplicate_text_rows": duplicate_text_rows,
                "duplicate_text_extra_rows": duplicate_text_extra_rows,
                "cross_split_duplicate_text_rows": cross_split_duplicate_text_rows,
                "invalid_label_values": invalid_label_values,
                "emotions_mismatch_rows": emotions_mismatch_rows,
            })

    # Count rows in each split that share normalized text with another split.
    split_duplicate_counts: dict[tuple[str, str], int] = defaultdict(int)
    for lang, text_to_splits in seen_texts.items():
        for text, splits in text_to_splits.items():
            if len(splits) > 1:
                for split in splits:
                    split_duplicate_counts[(lang, split)] += 1
    for row in split_rows:
        row["cross_split_duplicate_text_rows"] = split_duplicate_counts[(row["lang"], row["split"])]

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_df = pd.DataFrame(split_rows)
    labels_df = pd.DataFrame(label_rows)
    splits_df.to_csv(output_dir / "audit_splits.csv", index=False, encoding="utf-8-sig")
    labels_df.to_csv(output_dir / "audit_labels.csv", index=False, encoding="utf-8-sig")
    return splits_df, labels_df


def main() -> None:
    """解析命令行参数并运行对应任务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=resolve_path("data/raw/brighter"))
    parser.add_argument("--output-dir", type=Path, default=resolve_path("reports/brighter"))
    args = parser.parse_args()
    splits, labels = inspect(args.input_dir, args.output_dir)
    print(f"Audited {len(splits)} language/split combinations and {len(labels)} label distributions.")
    print(f"Reports written to {args.output_dir.resolve()}")
    missing = splits[splits["status"] == "missing"]
    if not missing.empty:
        print(f"Missing splits: {len(missing)}")
    print(f"Rows: {int(splits['rows'].sum())}; label/emotions mismatches: {int(splits['emotions_mismatch_rows'].sum())}")


if __name__ == "__main__":
    main()
