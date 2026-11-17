"""测试文件：检查项目功能是否符合预期。"""

from pathlib import Path
import logging

import numpy as np
import pytest

from emotion_steering.data.brighter import EMOTIONS, LANGUAGES, load_brighter_local
from emotion_steering.data.parallel import validate_parallel_records
from emotion_steering.data.registry import DATASETS, load_dataset
from emotion_steering.utils.paths import resolve_path

DATA_ROOT = resolve_path("data/processed/brighter")


def _check_result(result):
    """验证 check result行为。"""
    data, labels, concepts, dataset_type = result
    assert dataset_type == "multitask"
    assert len(data) == len(labels) == len(concepts) > 0
    for task, y in zip(data, labels):
        assert isinstance(y, np.ndarray)
        assert y.ndim == 2 and y.shape[1] == 1 and len(task) == y.shape[0]
        assert set(y.flatten()) == {0, 1}
        for sample in task:
            assert set(sample) >= {"system", "user", "assistant"}
            assert sample["system"] == "" and sample["assistant"] == ""
            assert sample["user"].strip()


def test_all_registered_languages_load_local_data():
    """验证all registered languages load local data行为。"""
    for language in LANGUAGES:
        _check_result(load_dataset("brighter", language=language, data_root=DATA_ROOT))
    assert set(DATASETS) == {"brighter"}


def test_emotion_selection_and_language_isolation():
    """验证emotion selection and language isolation行为。"""
    eng = load_brighter_local("eng", emotions=["joy"], data_root=DATA_ROOT)
    chn = load_brighter_local("chn", emotions=["joy"], data_root=DATA_ROOT)
    _check_result(eng)
    _check_result(chn)
    assert eng[2] == chn[2] == ["joy"]
    assert eng[0][0] != chn[0][0]
    assert eng[1][0].tolist() == chn[1][0].tolist()


def test_empty_emotion_task_is_skipped_with_reason(caplog):
    """验证empty emotion task is skipped with reason行为。"""
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="No BRIGHTER tasks"):
            load_brighter_local("eng", emotions=["disgust"], data_root=DATA_ROOT)
    assert "language=eng emotion=disgust" in caplog.text
    assert "Skipping BRIGHTER task" in caplog.text


def test_parallel_data_rejects_duplicate_scenario_language_pair():
    """验证平行数据会拒绝重复的情境语言组合。"""
    records = [
        {"sample_id": "x1", "scenario_id": "s1", "language": "eng", "user": "hello",
         "split": "test", "translation_reviewed": True},
        {"sample_id": "x2", "scenario_id": "s1", "language": "eng", "user": "hi",
         "split": "test", "translation_reviewed": True},
    ]
    with pytest.raises(ValueError, match="Duplicate scenario/language"):
        validate_parallel_records(records, ["eng"])
