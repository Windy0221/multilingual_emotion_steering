"""加载分词器和支持 Steering 的语言模型。"""

from __future__ import annotations


def load_tokenizer(model_name: str, **kwargs):
    """加载快速分词器并设置填充词元。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Model loading requires transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
    if not tokenizer.is_fast:
        raise RuntimeError("Multilingual user-token extraction requires a fast tokenizer with offset mappings")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _hooked_model_class(model_type: str):
    """按模型类型选择带钩子的模型类。"""
    if model_type == "llama":
        from .backends.llama import SteeredLlamaForCausalLM
        return SteeredLlamaForCausalLM
    if model_type == "qwen3":
        from .backends.qwen3 import SteeredQwen3ForCausalLM
        return SteeredQwen3ForCausalLM
    raise ValueError(f"No hooked model wrapper for model_type={model_type!r}")


def load_model(model_name: str, hooked: bool = False, **kwargs):
    """加载模型并设置设备和权重精度。"""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("Model loading requires transformers") from error
    dtype = kwargs.get("torch_dtype")
    if isinstance(dtype, str):
        if not hasattr(__import__("torch"), dtype):
            raise ValueError(f"Unsupported torch_dtype: {dtype}")
        import torch
        kwargs["torch_dtype"] = getattr(torch, dtype)
    if hooked:
        config = AutoConfig.from_pretrained(model_name)
        model_class = _hooked_model_class(config.model_type)
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(model_name, **kwargs)
    model.eval()
    return model
