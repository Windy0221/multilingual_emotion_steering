"""定义评价器接口和评分格式校验。"""

from __future__ import annotations

EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")


class GenerationEvaluator:
    def evaluate(self, record: dict) -> dict:
        """使用评价模型为一条生成记录评分。"""
        raise NotImplementedError


def validate_score(score: dict, record: dict) -> dict:
    """检查评分字段和分数范围。"""
    emotions = score.get("emotion_scores")
    if not isinstance(emotions, dict) or set(emotions) != set(EMOTIONS):
        raise ValueError(f"emotion_scores must contain exactly {EMOTIONS}")
    for name, value in emotions.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
            raise ValueError(f"{name} intensity must be an integer from 0 to 3")
    for key, low, high in (("relevance", 1, 5), ("fluency", 1, 5)):
        if key not in score or isinstance(score[key], bool) or not isinstance(score[key], int) or not low <= score[key] <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    if "language_correct" not in score or not isinstance(score["language_correct"], bool):
        raise ValueError("language_correct must be boolean")
    return {"sample_id": record.get("sample_id", record.get("id")),
            "target_language": record.get("target_language", record.get("language")),
            "target_emotion": record.get("target_emotion", record.get("emotion")),
            **score, "scoring_status": "scored"}
