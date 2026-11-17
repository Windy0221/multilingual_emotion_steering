"""准备 SFT/DPO 数据并训练 LoRA 适配器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emotion_steering.utils.io import load_jsonl


def _chat_kwargs(config):
    """返回训练阶段统一使用的聊天模板参数。"""
    return dict(config.get("chat_template", {"enable_thinking": False}))


def validate_sft_records(records):
    """检查监督微调数据格式。"""
    if not records:
        raise ValueError("SFT requires at least one response-supervision record")
    for index, row in enumerate(records):
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("user", "assistant")):
            raise ValueError(f"SFT record {index} requires nonempty user and assistant text")
    return records


def validate_dpo_records(records):
    """检查偏好数据格式和回复差异。"""
    if not records:
        raise ValueError("DPO requires at least one preference pair")
    for index, row in enumerate(records):
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("user", "chosen", "rejected")):
            raise ValueError(f"DPO record {index} requires nonempty user, chosen, and rejected text")
        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError(f"DPO record {index} has identical chosen and rejected responses")
    return records


def assistant_only_features(prompt_ids, full_ids, max_length=None):
    """构造仅对助手回复计算损失的特征。"""
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt tokens are not a prefix of the full assistant conversation")
    if max_length is not None:
        full_ids = full_ids[:int(max_length)]
    prompt_length = min(len(prompt_ids), len(full_ids))
    if prompt_length >= len(full_ids):
        raise ValueError("Assistant response has no trainable tokens after truncation")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * prompt_length + full_ids[prompt_length:]}


class AssistantOnlyCollator:
    """Right-pad sequences and ignore all padded and prompt positions in the loss."""
    def __init__(self, pad_token_id):
        """初始化当前对象及其配置。"""
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features):
        """处理一次模型前向调用。"""
        import torch
        width = max(len(row["input_ids"]) for row in features)
        input_ids = torch.full((len(features), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), width), dtype=torch.long)
        labels = torch.full((len(features), width), -100, dtype=torch.long)
        for index, row in enumerate(features):
            length = len(row["input_ids"])
            input_ids[index, :length] = torch.as_tensor(row["input_ids"], dtype=torch.long)
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.as_tensor(row["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def prepare_dpo_records(records, tokenizer, chat_template_kwargs=None):
    """将偏好记录转换为 DPO 所需格式。"""
    validate_dpo_records(records)
    converted = []
    for row in records:
        messages = []
        if row.get("system"):
            messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": row["user"]})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              **(chat_template_kwargs or {"enable_thinking": False}))
        converted.append({"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]})
    return converted



from .base import MethodArtifact, SteeringMethod


class SFTSteering(SteeringMethod):
    name = "sft"
    artifact_type = "peft_adapter"

    def fit(self, model, data, config):
        """根据数据拟合当前 Steering 方法。"""
        records = validate_sft_records(data)
        trainer = config.get("trainer")
        if trainer is None:
            raise RuntimeError("SFT data is valid, but training requires a configured trainer/model runtime")
        result = trainer(model=model, records=records, config=config)
        return MethodArtifact(self.name, self.artifact_type, result, {"record_count": len(records)})

    def apply(self, model, artifact, config):
        """将当前方法应用到模型或生成配置。"""
        if artifact.artifact_type != self.artifact_type:
            raise TypeError("SFT requires a peft_adapter artifact")
        return artifact.payload



from .base import MethodArtifact, SteeringMethod


class DPOSteering(SteeringMethod):
    name = "dpo"
    artifact_type = "peft_adapter"

    def fit(self, model, data, config):
        """根据数据拟合当前 Steering 方法。"""
        records = validate_dpo_records(data)
        trainer = config.get("trainer")
        if trainer is None:
            raise RuntimeError("DPO data is valid, but training requires a configured trainer/model runtime")
        result = trainer(model=model, records=records, config=config)
        return MethodArtifact(self.name, self.artifact_type, result, {"pair_count": len(records)})

    def apply(self, model, artifact, config):
        """将当前方法应用到模型或生成配置。"""
        if artifact.artifact_type != self.artifact_type:
            raise TypeError("DPO requires a peft_adapter artifact")
        return artifact.payload






def run(config):
    """根据配置执行模型训练或实验。"""
    method = str(config.get("method", "")).lower()
    if method not in {"sft", "dpo"}:
        raise ValueError("Training config method must be sft or dpo")
    records = load_jsonl(config["data_path"])
    records = validate_sft_records(records) if method == "sft" else validate_dpo_records(records)
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as error:
        raise RuntimeError("SFT/DPO training requires transformers and peft; install them before training") from error
    if method == "dpo":
        try:
            from trl import DPOConfig, DPOTrainer
        except ImportError as error:
            raise RuntimeError("DPO training requires trl") from error
        return _train_dpo(config, records, DPOConfig, DPOTrainer, LoraConfig, get_peft_model,
                          AutoModelForCausalLM, AutoTokenizer)
    return _train_sft(config, records, LoraConfig, get_peft_model, AutoModelForCausalLM,
                      AutoTokenizer, Trainer, TrainingArguments)


def _base(config, auto_model, auto_tokenizer, LoraConfig, get_peft_model):
    """加载适配器训练所需的基础模型。"""
    model_name = config["model"]
    dtype = config.get("torch_dtype", "bfloat16")
    if isinstance(dtype, str):
        import torch
        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported torch_dtype: {dtype}")
        dtype = getattr(torch, dtype)
    model = auto_model.from_pretrained(model_name, device_map=config.get("device", "auto"), torch_dtype=dtype)
    tokenizer = auto_tokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = get_peft_model(model, LoraConfig(
        r=int(config.get("lora_r", 16)), lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)), task_type="CAUSAL_LM",
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])))
    return model, tokenizer


def _save_training_metadata(config, output_dir):
    """保存适配器对应的模型、精度和聊天模板设置。"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata = {key: config.get(key) for key in ("model", "torch_dtype", "chat_template", "max_length",
                                                  "method", "target_emotion", "training_language")}
    (path / "steering_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_sft(config, records, LoraConfig, get_peft_model, AutoModel, AutoTokenizer, Trainer, TrainingArguments):
    """训练 LoRA 监督微调适配器。"""
    from datasets import Dataset
    model, tokenizer = _base(config, AutoModel, AutoTokenizer, LoraConfig, get_peft_model)
    def encode(row):
        """将对话记录转换为输入词元。"""
        messages = []
        if row.get("system"): messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": row["user"]})
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   **_chat_kwargs(config))
        messages.append({"role": "assistant", "content": row["assistant"]})
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                 **_chat_kwargs(config))
        return assistant_only_features(prompt_ids, full_ids, max_length=int(config.get("max_length", 2048)))
    dataset = Dataset.from_list(records).map(encode, remove_columns=list(records[0]))
    args = TrainingArguments(output_dir=config["output_dir"], num_train_epochs=float(config.get("epochs", 1)),
                             per_device_train_batch_size=int(config.get("batch_size", 1)),
                             learning_rate=float(config.get("learning_rate", 2e-5)),
                             logging_steps=int(config.get("logging_steps", 10)), save_strategy="epoch", report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
                      data_collator=AssistantOnlyCollator(tokenizer.pad_token_id))
    result = trainer.train(); trainer.save_model(config["output_dir"]); tokenizer.save_pretrained(config["output_dir"])
    _save_training_metadata(config, config["output_dir"])
    return result


def _train_dpo(config, records, DPOConfig, DPOTrainer, LoraConfig, get_peft_model, AutoModel, AutoTokenizer):
    """训练 LoRA 偏好优化适配器。"""
    from datasets import Dataset
    model, tokenizer = _base(config, AutoModel, AutoTokenizer, LoraConfig, get_peft_model)
    converted_records = prepare_dpo_records(records, tokenizer, _chat_kwargs(config))
    args = DPOConfig(output_dir=config["output_dir"], num_train_epochs=float(config.get("epochs", 1)),
                     per_device_train_batch_size=int(config.get("batch_size", 1)),
                     learning_rate=float(config.get("learning_rate", 2e-6)), report_to=[])
    import inspect
    processing_arg = "processing_class" if "processing_class" in inspect.signature(DPOTrainer).parameters else "tokenizer"
    trainer = DPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(converted_records),
                         **{processing_arg: tokenizer})
    result = trainer.train(); trainer.save_model(config["output_dir"]); tokenizer.save_pretrained(config["output_dir"])
    _save_training_metadata(config, config["output_dir"])
    return result
