"""提取用户文本词元对应的模型激活。"""

from __future__ import annotations

import torch


HOOK_LOCATION = 7
HOOK_NAME_TEMPLATE = "model.layers.{layer}.hook_after_mlp_hs"


def determine_generation_phase(past_key_values, input_length=None, prompt_length=None):
    """Classify forward calls from generation cache state and prompt progress."""
    has_past = past_key_values is not None
    if has_past and hasattr(past_key_values, "get_seq_length"):
        has_past = int(past_key_values.get_seq_length()) > 0
    if has_past:
        return "generation"
    if prompt_length is not None and input_length is not None and int(input_length) > int(prompt_length):
        return "generation"
    return "prefill"


def render_user_messages(rows, tokenizer, chat_template_kwargs=None):
    """套用聊天模板并定位用户原文。"""
    texts = []
    spans = []
    for row in rows:
        messages = [{"role": "user", "content": str(row["text"])}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 **(chat_template_kwargs or {}))
        start = rendered.rfind(str(row["text"]))
        if start < 0:
            raise ValueError("Tokenizer chat template did not preserve BRIGHTER user text")
        texts.append(rendered)
        spans.append((start, start + len(str(row["text"]))))
    return texts, spans


@torch.inference_mode()
def extract_user_token_activations(rows, tokenizer, model, batch_size=4, device=None, pooling="mean",
                                   chat_template_kwargs=None):
    """Capture PsySET block-output hooks and pool only user-text token positions."""
    if pooling not in {"mean", "last"}:
        raise ValueError("pooling must be mean or last")
    blocks = getattr(getattr(model, "model", None), "layers", None)
    if blocks is None:
        text_model = getattr(getattr(model, "model", None), "language_model", None)
        blocks = getattr(text_model, "layers", None)
    if blocks is None:
        raise TypeError("Hooked PsySET model does not expose model.layers")
    hook_modules = []
    for layer_index, block in enumerate(blocks):
        hook_module = getattr(block, "hook_after_mlp_hs", None)
        if hook_module is None:
            raise TypeError(f"PsySET block {layer_index} has no hook_after_mlp_hs module")
        hook_modules.append(hook_module)
    all_layers: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(hook_modules))}
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, hook_module in enumerate(hook_modules):
        def capture(_module, _inputs, output, index=layer_index):
            """保存当前层前向计算得到的激活。"""
            captured[index] = output
        handles.append(hook_module.register_forward_hook(capture))
    texts, spans = render_user_messages(rows, tokenizer, chat_template_kwargs)
    model_device = device or getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    try:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_spans = spans[start : start + batch_size]
            encoded = tokenizer(batch_texts, padding=True, return_tensors="pt", return_offsets_mapping=True,
                                truncation=False, add_special_tokens=False)
            offsets = encoded.pop("offset_mapping")
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            captured.clear()
            model(**encoded, use_cache=False)
            if set(captured) != set(range(len(hook_modules))):
                raise RuntimeError(f"Expected every hook_after_mlp_hs layer to fire; observed layers {sorted(captured)}")
            for item_index, (char_start, char_end) in enumerate(batch_spans):
                token_offsets = offsets[item_index]
                selected_cpu = ((token_offsets[:, 0] < char_end) & (token_offsets[:, 1] > char_start))
                selected_cpu &= encoded["attention_mask"][item_index].detach().cpu().bool()
                if not selected_cpu.any():
                    raise ValueError(f"No token offsets map to user text for row {start + item_index}")
                for layer_index, layer_output in captured.items():
                    selected = selected_cpu.to(layer_output.device)
                    selected_states = layer_output[item_index, selected]
                    pooled = selected_states.mean(dim=0) if pooling == "mean" else selected_states[-1]
                    all_layers[layer_index].append(pooled.float().cpu())
    finally:
        for handle in handles:
            handle.remove()
    return {layer: torch.stack(values) for layer, values in all_layers.items()}


