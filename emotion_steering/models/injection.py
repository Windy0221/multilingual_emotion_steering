"""管理 Steering 钩子和向量注入。"""

from __future__ import annotations

import torch

EXTRACTION_LOCATIONS = {7: "model.layers.[LID].hook_after_mlp_hs"}


def name_to_loc_and_layer(name: str) -> tuple[int, int]:
    """解析钩子对应的层号和位置。"""
    if "model.layers." not in name:
        raise ValueError(f"Hook name does not identify a transformer layer: {name}")
    layer = int(name.split("model.layers.", 1)[1].split(".", 1)[0])
    suffix_to_location = {value.split(".")[-1]: key for key, value in EXTRACTION_LOCATIONS.items()}
    suffix = name.rsplit(".", 1)[-1]
    if suffix not in suffix_to_location:
        raise ValueError(f"Hook name does not map to a supported extraction location: {name}")
    return suffix_to_location[suffix], layer


def get_steer_fn(steer_vector, steer_layers, steer_locs, tokens_slice, hs_size, mode, renormalize):
    """创建向目标层注入向量的函数。"""
    steer_layers = list(map(int, steer_layers))
    steer_locs = list(map(int, steer_locs))
    if any(location not in EXTRACTION_LOCATIONS or location == 10 for location in steer_locs):
        raise ValueError(f"Invalid steering hook locations: {steer_locs}")
    if mode not in {"add", "replace"}:
        raise ValueError("mode must be add or replace")
    if steer_vector.ndim == 3:
        steer_vector = steer_vector.unsqueeze(2)
    expected = (len(steer_layers), len(steer_locs), 1, int(hs_size))
    if tuple(steer_vector.shape) != expected:
        raise ValueError(f"steer_vector shape must be {expected}, got {tuple(steer_vector.shape)}")
    steer_vector = steer_vector.unsqueeze(0)
    target_names = {
        EXTRACTION_LOCATIONS[location].replace("[LID]", str(layer))
        for layer in steer_layers for location in steer_locs
    }

    def steer_fn(activations, hook):
        """对目标层激活执行向量干预。"""
        if getattr(hook, "name", None) not in target_names:
            return activations
        location, layer = name_to_loc_and_layer(hook.name)
        layer_index, location_index = steer_layers.index(layer), steer_locs.index(location)
        selected = activations[:, tokens_slice, :]
        vector = steer_vector[:, layer_index, location_index, tokens_slice].to(
            device=activations.device, dtype=activations.dtype
        )
        original_norm = selected.norm(dim=-1, keepdim=True) if renormalize else None
        changed = vector if mode == "replace" else selected + vector
        if renormalize:
            changed = changed / changed.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(changed.dtype).eps) * original_norm
        result = activations.clone()
        result[:, tokens_slice, :] = changed.to(dtype=activations.dtype)
        return result

    return steer_fn


def apply_activation_vector(activations, vector, coefficient=1.0, mode="add", renormalize=False, token_indices=None):
    """按指定方式将向量注入激活。"""
    if activations.ndim != 3:
        raise ValueError("activations must have [batch, sequence, hidden] shape")
    vector = torch.as_tensor(vector, device=activations.device, dtype=activations.dtype)
    if vector.numel() != activations.shape[-1]:
        raise ValueError("vector hidden size does not match activation hidden size")
    if mode not in {"add", "replace"}:
        raise ValueError("mode must be add or replace")
    result = activations.clone()
    indices = list(range(activations.shape[1])) if token_indices is None else [i % activations.shape[1] for i in token_indices]
    original = result[:, indices, :]
    scaled = vector.reshape(1, 1, -1) * float(coefficient)
    changed = original + scaled if mode == "add" else scaled.expand_as(original)
    if renormalize:
        old_norm = original.norm(dim=-1, keepdim=True)
        new_norm = changed.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(changed.dtype).eps)
        changed = changed / new_norm * old_norm
    result[:, indices, :] = changed
    return result


def scale_steering_vectors(vectors, layers, coefficient=1.0, normalize=False, device=None, dtype=None):
    """缩放并可选归一化向量。"""
    selected = torch.stack([vectors[int(layer)] for layer in layers])
    if normalize:
        norms = selected.norm(dim=-1, keepdim=True)
        selected = torch.where(norms > 0, selected / norms.clamp_min(1e-12), selected)
    selected = selected * float(coefficient)
    if device is not None or dtype is not None:
        selected = selected.to(device=device, dtype=dtype)
    return selected


def validate_vector_hook_locations(metadata, requested):
    """检查向量提取和注入位置是否一致。"""
    locations = list(requested)
    expected = list(metadata.get("hook_locations", [7]))
    if locations != expected:
        raise ValueError("Requested hook locations do not match the vector extraction metadata")
    if locations != [7]:
        raise ValueError("BRIGHTER vectors are extracted from transformer block outputs; only PsySET hook location 7 is compatible")
    return locations


def gate_intervention_stage(hook_fn, stage="both", phase_getter=None, target_names=None):
    """限制向量注入发生的生成阶段。"""
    if stage not in {"prefill", "generation", "both"}:
        raise ValueError("stage must be prefill, generation, or both")
    targets = set(target_names or ())
    call_counts = {"prefill": {}, "generation": {}}
    observed_counts = {"prefill": {}, "generation": {}}
    forward_phase_counts = {"prefill": 0, "generation": 0}

    def gated(activations, hook):
        """根据生成阶段决定是否执行干预。"""
        phase = phase_getter() if phase_getter is not None else None
        if phase not in {"prefill", "generation"}:
            phase = "prefill" if activations.shape[1] > 1 else "generation"
        forward_phase_counts[phase] += 1
        hook_name = getattr(hook, "name", None)
        if targets and hook_name not in targets:
            return activations
        if hook_name is not None:
            observed_counts[phase][hook_name] = observed_counts[phase].get(hook_name, 0) + 1
        if stage != "both" and stage != phase:
            return activations
        if hook_name is not None:
            call_counts[phase][hook_name] = call_counts[phase].get(hook_name, 0) + 1
        return hook_fn(activations, hook)

    gated.call_counts = call_counts
    gated.observed_counts = observed_counts
    gated.forward_phase_counts = forward_phase_counts
    return gated
