"""测试文件：检查项目功能是否符合预期。"""

import json

import pandas as pd

from emotion_steering.data.inspect import EMOTIONS
from emotion_steering.data.preprocess import preprocess


def _write_source(root, lang="eng"):
    """验证 write source行为。"""
    for split in ("train", "dev", "test"):
        folder = root / lang
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, label in enumerate((1, 0, None)):
            row = {"id": f"{lang}_{split}_{index}", "text": f"{split} sample {index}", "emotions": []}
            for emotion in EMOTIONS:
                row[emotion] = label if emotion == "joy" else (0 if label is not None else None)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(folder / f"{split}-00000-of-00001.parquet", index=False)


def test_preprocess_preserves_unknowns_and_samples_train_only(tmp_path):
    """验证preprocess preserves unknowns and samples train only行为。"""
    source, output = tmp_path / "source", tmp_path / "processed"
    _write_source(source)
    manifest = preprocess(source, output, per_class=1, seed=42)

    normalized = pd.read_parquet(output / "normalized/eng/train.parquet")
    assert normalized["joy"].isna().sum() == 1
    task = pd.read_parquet(output / "binary_train/eng/joy.parquet")
    assert task["label"].tolist().count(1) == 1
    assert task["label"].tolist().count(0) == 1
    assert not task["id"].str.contains("dev|test").any()
    assert set(manifest["languages"]["eng"]["emotions"]["joy"]["selected_ids"]) == {"0", "1"}


def test_sampling_manifest_is_reproducible(tmp_path):
    """验证sampling manifest is reproducible行为。"""
    source = tmp_path / "source"
    _write_source(source)
    first = preprocess(source, tmp_path / "out1", per_class=1, seed=42)
    second = preprocess(source, tmp_path / "out2", per_class=1, seed=42)
    assert first["languages"]["eng"]["emotions"] == second["languages"]["eng"]["emotions"]
    assert json.loads((tmp_path / "out1/sampling_manifest.json").read_text()) == first
