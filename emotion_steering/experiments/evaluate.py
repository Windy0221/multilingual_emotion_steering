"""运行分类或生成回复评价并保存结果。"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from emotion_steering.evaluation.metrics import summarize_scores
from emotion_steering.evaluation.llm_judge import OpenAIJudge


def evaluate_emotion_classification(records):
    """根据标签和预测结果计算分类指标。"""
    if not records:
        raise ValueError("Classification evaluation needs labeled records")
    y_true = [int(row["label"]) for row in records]
    y_pred = [int(row["prediction"]) for row in records]
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    return {"task": "brighter_emotion_classification", "n": len(records),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "class_report": report}


def evaluate_generation(records, evaluator=None, aggregation=None):
    """对生成回复评分并汇总结果。"""
    if not records:
        raise ValueError("Generation evaluation needs saved generation records")
    scores = []
    for record in records:
        if evaluator is None:
            scores.append({**record, "scoring_status": "not_scored"})
        else:
            result = evaluator.evaluate(record) if hasattr(evaluator, "evaluate") else evaluator(record)
            scores.append({**record, **result})
    summary = summarize_scores(scores, **(aggregation or {}))
    return {"task": "generation", "n": len(records), "scoring_status": "scored" if evaluator else "not_scored",
            "scores": scores, "summary": summary}


def evaluate_file(input_path, output_path, task="classification", evaluator_config=None, aggregation=None):
    """读取输入、执行评价并保存结果。"""
    source = Path(input_path)
    if task == "classification":
        frame = pd.read_csv(source)
        result = evaluate_emotion_classification(frame.to_dict("records"))
    elif task == "generation":
        sources = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
        records = []
        for file in sources:
            records.extend(json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
        evaluator = None
        evaluator_config = evaluator_config or {}
        if evaluator_config.get("type") == "llm" and evaluator_config.get("model"):
            if evaluator_config.get("provider", "openai") != "openai":
                raise ValueError("Only provider=openai is currently supported")
            evaluator = OpenAIJudge(evaluator_config["model"], evaluator_config.get("temperature", 0),
                                   evaluator_config.get("api_key"))
        result = _evaluate_generation_resumable(records, evaluator, evaluator_config, output_path, aggregation)
    else:
        raise ValueError("task must be classification or generation")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(output_path).with_suffix(Path(output_path).suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output_path)
    if task == "generation":
        comparisons = result.get("summary", {}).get("paired_comparisons", [])
        comparison_path = Path(output_path).with_suffix(".comparisons.csv")
        pd.DataFrame(comparisons).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    return result


def _evaluate_generation_resumable(records, evaluator, evaluator_config, output_path, aggregation):
    """逐条保存评分，跳过已完成项目并保留失败状态。"""
    config = evaluator_config or {}
    evaluator_fingerprint = hashlib.sha256(json.dumps(
        {"provider": config.get("provider"), "model": config.get("model"), "temperature": config.get("temperature", 0),
         "template_version": config.get("template_version", "emotion-rubric-v1")},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    score_path = Path(output_path).with_suffix(".scores.jsonl")
    score_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if score_path.exists():
        for line in score_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item.get("experiment_id"), item.get("sample_id"), item.get("evaluator_fingerprint"),
                   item.get("record_fingerprint"))
            completed[key] = item
    seen = set()
    with score_path.open("a", encoding="utf-8") as stream:
        try:
            from tqdm.auto import tqdm
            record_iterator = tqdm(records, desc="回复评分", unit="条")
        except ImportError:
            record_iterator = records
        for record in record_iterator:
            sample_id = record.get("sample_id", record.get("id"))
            if sample_id is None:
                raise ValueError("Every generation record needs sample_id or id for resumable evaluation")
            record_payload = {key: record.get(key) for key in ("user", "response", "target_language", "target_emotion", "emotion")}
            record_fingerprint = hashlib.sha256(json.dumps(record_payload, ensure_ascii=False, sort_keys=True,
                                                             separators=(",", ":")).encode("utf-8")).hexdigest()
            key = (record.get("experiment_id"), sample_id, evaluator_fingerprint, record_fingerprint)
            if key in seen:
                raise ValueError(f"Duplicate generation record identity: {key}")
            seen.add(key)
            previous_status = completed.get(key, {}).get("scoring_status")
            if previous_status == "scored" or (evaluator is None and previous_status == "not_scored"):
                continue
            if evaluator is None:
                result = {**record, "scoring_status": "not_scored"}
            else:
                try:
                    score = evaluator.evaluate(record) if hasattr(evaluator, "evaluate") else evaluator(record)
                    result = {**record, **score, "scoring_status": "scored"}
                except Exception as error:
                    result = {**record, "scoring_status": "error", "scoring_error": str(error)}
            result["evaluator_fingerprint"] = evaluator_fingerprint
            result["evaluator_model"] = config.get("model")
            result["evaluator_template_version"] = config.get("template_version", "emotion-rubric-v1")
            result["evaluation_key"] = list(key)
            result["record_fingerprint"] = record_fingerprint
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            completed[key] = result
    output_scores = []
    for row in records:
        payload = {key: row.get(key) for key in ("user", "response", "target_language", "target_emotion", "emotion")}
        content_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                 separators=(",", ":")).encode("utf-8")).hexdigest()
        key = (row.get("experiment_id"), row.get("sample_id", row.get("id")), evaluator_fingerprint, content_hash)
        output_scores.append(completed[key])
    summary = summarize_scores(output_scores, **(aggregation or {}))
    status = "scored" if evaluator and not any(row.get("scoring_status") == "error" for row in output_scores) else (
        "partial" if evaluator else "not_scored")
    return {"task": "generation", "n": len(records), "scoring_status": status,
            "scores_path": str(score_path), "scores": output_scores, "summary": summary}
