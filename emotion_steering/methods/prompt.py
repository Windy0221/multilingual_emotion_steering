"""构造不同类型的情绪引导提示。"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .base import MethodArtifact, SteeringMethod


def _load_prompt_resources(config):
    """加载本地化提示模板和情绪描述资源。"""
    path = Path(config.get("prompt_templates_path", "configs/prompts/templates.yaml"))
    if not path.is_absolute():
        from emotion_steering.utils.paths import resolve_path
        path = resolve_path(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_few_shot_examples(config, language, emotion, count):
    """按示例语言、目标情绪和固定数量选择示例。"""
    path = Path(config.get("few_shot_examples_path", "configs/prompts/few_shot_examples.jsonl"))
    if not path.is_absolute():
        from emotion_steering.utils.paths import resolve_path
        path = resolve_path(path)
    if not path.is_file():
        raise ValueError(f"Few-shot example file does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    identifiers = [row.get("example_id") for row in rows]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Few-shot library requires unique, nonempty example_id values")
    required_fields = ("scenario_id", "language", "target_emotion", "split", "user", "assistant")
    for index, row in enumerate(rows):
        missing = [key for key in required_fields if not row.get(key)]
        if missing:
            raise ValueError(f"Few-shot example {index} is missing fields: {missing}")
        if row.get("translation_reviewed") is not True:
            raise ValueError(f"Few-shot example {row['example_id']} has not passed translation review")
    selected = [row for row in rows if row.get("language") == language and
                row.get("target_emotion") == emotion and row.get("split") == "few_shot"]
    selected.sort(key=lambda row: (int(row.get("order", 0)), row["example_id"]))
    if len(selected) < count:
        raise ValueError(f"Need {count} {language}/{emotion} few-shot examples; found {len(selected)}")
    scenario_ids = [row.get("scenario_id") for row in selected[:count]]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Few-shot examples must use distinct scenario_id values")
    return selected[:count]


class NoSteering(SteeringMethod):
    name = "none"
    artifact_type = "no_control"

    def fit(self, model, data, config):
        """根据数据拟合当前 Steering 方法。"""
        return MethodArtifact(self.name, self.artifact_type, None, {})

    def apply(self, model, artifact, config):
        """将当前方法应用到模型或生成配置。"""
        if artifact.artifact_type != self.artifact_type:
            raise TypeError("NoSteering requires a no_control artifact")
        return None


class PromptSteering(SteeringMethod):
    artifact_type = "system_prompt"

    def __init__(self, style: str = "zero_shot"):
        """初始化当前对象及其配置。"""
        if style not in {"zero_shot", "few_shot", "descriptive"}:
            raise ValueError(f"Unsupported prompt style: {style}")
        self.name = style
        self.style = style

    def fit(self, model, data, config):
        """根据数据拟合当前 Steering 方法。"""
        emotion = config.get("target_emotion", config.get("emotion"))
        language = config.get("target_language", config.get("language", "eng"))
        if not emotion:
            raise ValueError("Prompt steering requires a target_emotion")
        resources = _load_prompt_resources(config)
        prompt_config = config.get("prompt", {})
        instruction_language = prompt_config.get("instruction_language", "target")
        instruction_language = language if instruction_language == "target" else instruction_language
        locale = resources.get("languages", {}).get(instruction_language)
        emotion_label = resources.get("emotions", {}).get(instruction_language, {}).get(emotion)
        if not locale or not emotion_label:
            raise ValueError(f"No prompt template for language={instruction_language}, emotion={emotion}")
        prompt = ""
        if self.style in {"zero_shot", "descriptive"}:
            template_key = self.style
            replacements = {"emotion": emotion_label}
            if self.style == "descriptive":
                descriptions = resources.get("descriptions", {}).get(instruction_language, {})
                description = descriptions.get(emotion)
                if not description:
                    raise ValueError(f"No descriptive prompt for {instruction_language}/{emotion}")
                replacements["description"] = description
            prompt = locale[template_key].format(**replacements)
        else:
            examples_language = prompt_config.get("examples_language", "target")
            examples_language = language if examples_language == "target" else examples_language
            count = int(prompt_config.get("num_examples", 3))
            examples = _load_few_shot_examples(config, examples_language, emotion, count)
            prompt = locale["zero_shot"].format(emotion=emotion_label)
            for example in examples:
                prompt += f"\n\nUser: {example['user']}\nAssistant: {example['assistant']}"
        metadata = {"target_emotion": emotion, "target_language": language, "style": self.style,
                    "instruction_language": instruction_language, "prompt_config": prompt_config,
                    "example_ids": [item["example_id"] for item in examples] if self.style == "few_shot" else [],
                    "example_count": len(examples) if self.style == "few_shot" else 0,
                    "example_format": "text_role_labels_v1" if self.style == "few_shot" else None}
        return MethodArtifact(self.name, self.artifact_type, prompt, metadata)

    def apply(self, model, artifact, config):
        """将当前方法应用到模型或生成配置。"""
        if artifact.artifact_type != "system_prompt":
            raise TypeError("PromptSteering requires a system_prompt artifact")
        return artifact.payload


class ZeroShotSteering(PromptSteering):
    def __init__(self):
        """初始化当前对象及其配置。"""
        super().__init__("zero_shot")


class FewShotSteering(PromptSteering):
    def __init__(self):
        """初始化当前对象及其配置。"""
        super().__init__("few_shot")


class DescriptiveSteering(PromptSteering):
    def __init__(self):
        """初始化当前对象及其配置。"""
        super().__init__("descriptive")
