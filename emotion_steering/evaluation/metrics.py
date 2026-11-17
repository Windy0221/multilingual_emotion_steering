"""汇总评分并计算与基线的配对差值。"""

from __future__ import annotations

import json

import pandas as pd
import numpy as np

GROUP_COLUMNS = ("model", "run_id", "dataset_version", "method", "source_language",
                 "target_language", "target_emotion", "coefficient", "layers", "generation_signature")


def _json_value(value):
    """把统计结果中的空值和 NumPy 类型转为标准 JSON 值。"""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_scores(records: list[dict], compare_with_baseline: bool = True, baseline_method: str = "none") -> dict:
    """按实验条件汇总评分并计算基线差值。"""
    scored = [row for row in records if row.get("scoring_status") == "scored"]
    score_columns = [f"emotion_scores.{emotion}" for emotion in
                     ("anger", "disgust", "fear", "joy", "sadness", "surprise")]
    rows = []
    for row in scored:
        item = {key: row.get(key) for key in GROUP_COLUMNS}
        if isinstance(item.get("layers"), (list, tuple)):
            item["layers"] = json.dumps(item["layers"], separators=(",", ":"))
        item["sample_id"] = row.get("sample_id")
        item["scenario_id"] = row.get("scenario_id")
        item["comparison_id"] = row.get("comparison_id")
        item.update({f"emotion_scores.{key}": value for key, value in row["emotion_scores"].items()})
        item.update({key: row.get(key) for key in ("relevance", "fluency", "language_correct")})
        target = row.get("target_emotion", row.get("emotion"))
        target_score = row["emotion_scores"].get(target) if target else None
        item["target_emotion_score"] = target_score
        item["target_emotion_achieved"] = int(target_score >= 2) if target_score is not None else None
        rows.append(item)
    frame = pd.DataFrame(rows)
    group_columns = [key for key in GROUP_COLUMNS if key in frame.columns and frame[key].notna().any()]
    groups = []
    if not frame.empty:
        metrics = [key for key in score_columns + ["target_emotion_score", "target_emotion_achieved",
                                                   "relevance", "fluency", "language_correct"]
                   if key in frame and frame[key].notna().any()]
        if group_columns:
            aggregate = frame.groupby(group_columns, dropna=False)[metrics].agg(["mean", "count"]).round(4).reset_index()
            aggregate.columns = ["_".join(str(part) for part in column if part) if isinstance(column, tuple) else str(column)
                                 for column in aggregate.columns]
            groups = aggregate.astype(object).where(pd.notna(aggregate), None).to_dict("records")
    paired = []
    if compare_with_baseline and not frame.empty and "sample_id" in frame:
        identity = [key for key in ("model", "dataset_version", "generation_signature", "comparison_id", "sample_id",
                                    "scenario_id", "target_language", "target_emotion")
                    if key in frame.columns and frame[key].notna().any()]
        baseline_rows = frame[frame["method"] == baseline_method]
        if baseline_rows.duplicated(identity).any():
            raise ValueError("Duplicate baseline scores share the same comparison and sample identity")
        baseline = baseline_rows.set_index(identity)
        treatments = [key for key in ("model", "dataset_version", "method", "source_language",
                                      "coefficient", "layers", "generation_signature")
                      if key in frame.columns and frame[key].notna().any()]
        treatment_rows = frame[frame["method"] != baseline_method]
        treatment_groups = treatment_rows.groupby(treatments, dropna=False) if treatments and not treatment_rows.empty else []
        for values, subset in treatment_groups:
            if not isinstance(values, tuple): values = (values,)
            treatment = dict(zip(treatments, values))
            if subset.duplicated(identity).any():
                raise ValueError("Duplicate treatment scores share the same comparison and sample identity")
            candidates = subset.set_index(identity)
            shared = baseline.index.intersection(candidates.index)
            for metric in score_columns + ["target_emotion_score", "relevance", "fluency", "language_correct"]:
                if metric not in candidates or not len(shared):
                    continue
                deltas = candidates.loc[shared, metric].astype(float) - baseline.loc[shared, metric].astype(float)
                valid = deltas.dropna()
                mean = float(valid.mean()) if len(valid) else None
                margin = 1.96 * float(valid.std(ddof=1)) / (len(valid) ** 0.5) if len(valid) > 1 else None
                paired.append({**treatment, "baseline": baseline_method, "metric": metric,
                               "paired_n": int(len(valid)), "mean_delta": round(mean, 4) if mean is not None else None,
                               "ci95_low": round(mean - margin, 4) if margin is not None else None,
                               "ci95_high": round(mean + margin, 4) if margin is not None else None})
    return _json_value({"scored_count": len(scored), "not_scored_count": len(records) - len(scored),
                        "groups": groups, "paired_comparisons": paired})
