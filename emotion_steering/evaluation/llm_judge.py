"""调用独立 OpenAI 模型为生成回复评分。"""

from __future__ import annotations

import json

from .base import EMOTIONS, GenerationEvaluator, validate_score


class OpenAIJudge(GenerationEvaluator):
    """Optional OpenAI-compatible judge; requires an explicit model and API key."""
    def __init__(self, model: str, temperature: float = 0, api_key: str | None = None):
        """初始化当前对象及其配置。"""
        if not model:
            raise ValueError("An evaluator model must be configured before scoring")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("LLM judging requires the optional openai package") from error
        self.client = OpenAI(api_key=api_key)
        self.model, self.temperature = model, temperature

    def evaluate(self, record: dict) -> dict:
        # Deliberately omit steering method and coefficient to reduce evaluator bias.
        """使用评价模型为一条生成记录评分。"""
        payload = {key: record.get(key) for key in ("user", "response", "target_language")}
        prompt = ("Score this response without inferring a requested emotion or steering method. Return only JSON with emotion_scores (all six emotions, integers 0-3; "
                  "0 absent, 1 mild, 2 clear, 3 strong), language_correct (boolean), relevance (integer 1-5), "
                  "and fluency (integer 1-5). Treat the requested emotion and language as targets, not as proof.\n"
                  f"Input: {json.dumps(payload, ensure_ascii=False)}\nEmotions: {', '.join(EMOTIONS)}")
        response = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return validate_score(json.loads(response.choices[0].message.content), record)
