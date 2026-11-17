"""检查平行多语言生成测试集的覆盖、重复和数据泄漏。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from emotion_steering.utils.io import load_jsonl


def validate_parallel_records(records, required_languages, require_reviewed=True):
    """检查平行情境覆盖、样本编号和翻译审核状态。"""
    if not records:
        raise ValueError("Parallel evaluation data cannot be empty")
    ids, scenarios, coverage = set(), defaultdict(set), set()
    splits_by_scenario = defaultdict(set)
    for index, row in enumerate(records):
        required = ("sample_id", "scenario_id", "language", "user", "split")
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise ValueError(f"Record {index} is missing required fields: {missing}")
        if row["sample_id"] in ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        ids.add(row["sample_id"])
        pair_key = (row["scenario_id"], row["language"], row.get("replicate_id"))
        if pair_key in coverage:
            raise ValueError(f"Duplicate scenario/language pair: {pair_key}")
        scenario_key = (row["scenario_id"], row.get("replicate_id"))
        scenarios[scenario_key].add(row["language"])
        coverage.add(pair_key)
        splits_by_scenario[scenario_key].add(row["split"])
        if require_reviewed and row.get("translation_reviewed") is not True:
            raise ValueError(f"Record {row['sample_id']} has no approved translation review")
    errors = {scenario: sorted(set(required_languages) - languages)
              for scenario, languages in scenarios.items() if set(required_languages) - languages}
    if errors:
        raise ValueError(f"Parallel scenarios are missing languages: {errors}")
    leaked = {scenario: sorted(splits) for scenario, splits in splits_by_scenario.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Scenario leakage across data splits: {leaked}")
    return {"record_count": len(records), "scenario_count": len(scenarios),
            "languages": sorted(set.intersection(*(set(value) for value in scenarios.values())))}


def validate_parallel_file(path: str | Path, required_languages, require_reviewed=True):
    """读取平行数据文件并运行完整性检查。"""
    return validate_parallel_records(load_jsonl(path), required_languages, require_reviewed)
