"""测试文件：检查项目功能是否符合预期。"""

import json

import pytest

from emotion_steering.evaluation.base import EMOTIONS, validate_score
from emotion_steering.evaluation.metrics import summarize_scores
from emotion_steering.experiments.evaluate import evaluate_file, evaluate_generation


def scored(sample, method, joy):
    """验证scored行为。"""
    return {"sample_id": sample, "method": method, "source_language": "eng",
            "target_language": "chn", "emotion": "joy", "coefficient": 1,
            "emotion_scores": {emotion: joy if emotion == "joy" else 0 for emotion in EMOTIONS},
            "relevance": 4, "fluency": 5, "scoring_status": "scored"}


def test_score_validation_requires_all_emotions_and_valid_ranges():
    """验证score validation requires all emotions and valid ranges行为。"""
    record = {"sample_id": "a", "target_language": "chn", "emotion": "joy"}
    score = {"emotion_scores": {emotion: 2 for emotion in EMOTIONS}, "language_correct": True,
             "relevance": 4, "fluency": 5}
    assert validate_score(score, record)["scoring_status"] == "scored"
    score["emotion_scores"]["joy"] = 4
    with pytest.raises(ValueError):
        validate_score(score, record)


def test_paired_summary_uses_intersecting_sample_ids():
    """验证paired summary uses intersecting sample ids行为。"""
    summary = summarize_scores([scored("a", "none", 1), scored("b", "none", 0),
                                scored("a", "meandiff", 3), scored("c", "meandiff", 2)])
    json.dumps(summary)
    joy_delta = next(row for row in summary["paired_comparisons"] if row["metric"] == "emotion_scores.joy")
    assert joy_delta["paired_n"] == 1 and joy_delta["mean_delta"] == 2


def test_generation_without_judge_marks_not_scored_and_writes_file(tmp_path):
    """验证generation without judge marks not scored and writes file行为。"""
    source, output = tmp_path / "in.jsonl", tmp_path / "out.json"
    source.write_text(json.dumps({"sample_id": "a", "user": "hi", "response": "hello"}) + "\n")
    result = evaluate_file(source, output, "generation")
    assert result["scoring_status"] == "not_scored"
    assert result["scores"][0]["scoring_status"] == "not_scored"
    assert result["summary"]["not_scored_count"] == 1


def test_metrics_accept_layer_lists_and_pair_only_matching_models():
    records = []
    for model, baseline, steered in (("model-a", 0, 3), ("model-b", 3, 0)):
        base, treatment = scored("same", "none", baseline), scored("same", "meandiff", steered)
        for row in (base, treatment):
            row.update(model=model, dataset_version="data-v1", generation_signature="gen-v1",
                       layers=[16] if row["method"] != "none" else None)
        records.extend((base, treatment))
    summary = summarize_scores(records)
    deltas = [row for row in summary["paired_comparisons"] if row["metric"] == "target_emotion_score"]
    assert {row["model"]: row["mean_delta"] for row in deltas} == {"model-a": 3.0, "model-b": -3.0}
    json.dumps(summary)


def test_repeated_comparison_batches_are_paired_separately():
    records = []
    for comparison_id, base_score, steer_score in (("batch-1", 0, 2), ("batch-2", 3, 1)):
        for method, value in (("none", base_score), ("meandiff", steer_score)):
            row = scored("sample-1", method, value)
            row.update(model="model-a", dataset_version="d1", generation_signature="g1",
                       comparison_id=comparison_id, layers=[16] if method == "meandiff" else None)
            records.append(row)
    comparisons = summarize_scores(records)["paired_comparisons"]
    target = next(row for row in comparisons if row["metric"] == "target_emotion_score")
    assert target["paired_n"] == 2 and target["mean_delta"] == 0


def test_evaluation_saves_each_score_and_resumes_completed_records(tmp_path, monkeypatch):
    source = tmp_path / "generations.jsonl"
    output = tmp_path / "evaluation.json"
    record = {"sample_id": "s1", "experiment_id": "e1", "target_language": "chn",
              "target_emotion": "joy", "user": "hi", "response": "hello"}
    source.write_text(json.dumps(record) + "\n")

    class Evaluator:
        def __init__(self): self.calls = 0
        def evaluate(self, _record):
            self.calls += 1
            return {"emotion_scores": {emotion: (2 if emotion == "joy" else 0) for emotion in EMOTIONS},
                    "language_correct": True, "relevance": 4, "fluency": 5}

    evaluator = Evaluator()
    monkeypatch.setattr("emotion_steering.experiments.evaluate.OpenAIJudge", lambda *_args, **_kwargs: evaluator)
    config = {"type": "llm", "provider": "openai", "model": "judge-v1"}
    evaluate_file(source, output, "generation", config)
    evaluate_file(source, output, "generation", config)
    assert evaluator.calls == 1
    assert json.loads(output.read_text(encoding="utf-8"))["scoring_status"] == "scored"
    record["response"] = "a changed reply"
    source.write_text(json.dumps(record) + "\n")
    evaluate_file(source, output, "generation", config)
    assert evaluator.calls == 2
    assert json.loads(output.read_text(encoding="utf-8"))["scores"][0]["response"] == "a changed reply"
