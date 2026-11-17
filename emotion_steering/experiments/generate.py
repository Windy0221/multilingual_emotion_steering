"""执行 Steering 生成并保存模型回复。"""

from __future__ import annotations

import json
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from emotion_steering.utils.io import load_vector_artifact
from emotion_steering.models.injection import (gate_intervention_stage, get_steer_fn,
                                               scale_steering_vectors,
                                               validate_vector_hook_locations)
from emotion_steering.models.loader import load_model, load_tokenizer
from emotion_steering.methods.registry import create_method


def git_revision():
    """读取当前代码版本号。"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def experiment_signatures(config):
    """为实验条件和共同生成设置生成稳定签名。"""
    shared = {key: config.get(key) for key in ("model", "target_language", "chat_template", "tokenization",
                                                "generation", "generation_seed", "torch_dtype")}
    excluded = {"output_path", "output_dir", "run_id", "device"}
    condition = {key: value for key, value in config.items() if key not in excluded}
    for key in ("prompt_templates_path", "few_shot_examples_path", "vector_path", "adapter_path", "input_path"):
        value = config.get(key)
        if not value:
            continue
        path = Path(value)
        if path.is_file():
            condition[f"{key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if key in {"input_path", "prompt_templates_path"}:
                shared[f"{key}_sha256"] = condition[f"{key}_sha256"]
        elif path.is_dir():
            file_hashes = {child.relative_to(path).as_posix(): hashlib.sha256(child.read_bytes()).hexdigest()
                           for child in sorted(path.rglob("*")) if child.is_file()}
            condition[f"{key}_sha256"] = file_hashes
            if key == "input_path":
                shared[f"{key}_sha256"] = file_hashes
    def digest(value):
        """将配置映射为稳定的 SHA-256 标识。"""
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest(condition), digest(shared)


def dataset_fingerprint(input_path):
    """计算生成测试数据文件或目录的内容摘要。"""
    path = Path(input_path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        manifest = {item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
                    for item in sorted(path.rglob("*")) if item.is_file()}
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return None


def prepare_steering(model, record, config):
    """准备当前记录对应的提示或向量干预。"""
    method_name = config["method"]
    method = create_method(method_name)
    if method.artifact_type == "no_control":
        artifact = method.fit(model, record, config)
        return None, {"method": method_name, "metadata": artifact.metadata, "layers": None}
    if method.artifact_type == "system_prompt":
        prompt_config = {**record, **config}
        prompt_config["target_emotion"] = config.get("target_emotion", config.get("emotion"))
        prompt_config["target_language"] = config.get("target_language", record.get("language"))
        artifact = method.fit(model, record, prompt_config)
        return method.apply(model, artifact, config), artifact
    if method.artifact_type in {"activation_vector", "probe_and_vector"}:
        path = Path(config["vector_path"])
        target_emotion = config.get("target_emotion", config.get("emotion"))
        expected = {"model": config["model"], "source_language": config["source_language"],
                    "emotion": target_emotion, "method": method_name}
        vectors, metadata = load_vector_artifact(path, expected)
        layers = [int(layer) for layer in config.get("layers", [max(vectors)])]
        if any(layer not in vectors for layer in layers):
            raise ValueError(f"Requested layers {layers} are not all present in vector artifact")
        if metadata.get("hidden_size") != model.config.hidden_size:
            raise ValueError("Vector hidden size does not match loaded model")
        if metadata.get("model") != config["model"]:
            raise ValueError("Vector model metadata does not match current model")
        if metadata.get("chat_template", {}) != config.get("chat_template", {}):
            raise ValueError("Generation chat-template settings do not match vector extraction metadata")
        coeff = float(config.get("coefficient", 1.0))
        selected = scale_steering_vectors(vectors, layers, coefficient=coeff,
                                          normalize=config.get("normalize", False))
        if not hasattr(model, "set_steer_fn"):
            raise TypeError("The loaded model must provide PsySET set_steer_fn hook support")
        # PsySET's hook-location indices are model-family specific; callers must declare them.
        locations = list(config.get("hook_locations", [7]))
        validate_vector_hook_locations(metadata, locations)
        if len(layers) != len(set(layers)):
            raise ValueError("Intervention layers must be unique")
        steer_fn = get_steer_fn(selected[:, None, None, :], layers, locations,
                                 torch.tensor([-1]), model.config.hidden_size,
                                 config.get("intervention", "add"), config.get("renormalize", False))
        target_names = [f"model.layers.{layer}.hook_after_mlp_hs" for layer in layers]
        gated_steer_fn = gate_intervention_stage(
            steer_fn, config.get("stage", "both"),
            phase_getter=lambda: getattr(model, "_emotion_steering_generation_phase", None),
            target_names=target_names,
        )
        model.set_steer_fn(gated_steer_fn)
        model._emotion_steering_steer_fn = gated_steer_fn
        return None, {"method": method_name, "metadata": metadata, "layers": layers}
    if method.artifact_type == "peft_adapter":
        adapter_path = config.get("adapter_path")
        if not adapter_path:
            raise ValueError(f"{method_name.upper()} inference requires config.adapter_path")
        return None, {"method": method_name, "adapter_path": adapter_path, "layers": None}
    raise ValueError(f"Unsupported artifact type: {method.artifact_type}")


def validate_hook_execution(requested_phases, target_layers, call_counts, forward_phase_counts):
    """Report requested phases with no forward pass and reject missing hooks in observed phases."""
    skipped = {}
    for phase in requested_phases:
        if not forward_phase_counts or forward_phase_counts.get(phase, 0) == 0:
            skipped[phase] = "generation produced no forward pass in this phase"
            continue
        missing = [f"model.layers.{layer}.hook_after_mlp_hs" for layer in target_layers
                   if not call_counts or call_counts[phase].get(f"model.layers.{layer}.hook_after_mlp_hs", 0) == 0]
        if missing:
            raise RuntimeError(f"Intervention hooks did not execute during observed {phase} phase: {missing}")
    return skipped


def run_records(model, tokenizer, records, config, output_path):
    """生成回复并保存实验记录。"""
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = [row.get("sample_id", row.get("id")) for row in records]
    if any(sample_id is None for sample_id in sample_ids):
        raise ValueError("Every generation record must have sample_id or id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Input generation records contain duplicate sample_id values")
    run_metadata = {"run_id": config.get("run_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")),
                    "model": config["model"], "generation_config": config.get("generation", {}),
                    "chat_template": config.get("chat_template", {}),
                    "code_revision": git_revision(), "python": platform.python_version()}
    output_path = Path(output_path)
    experiment_id, generation_signature = experiment_signatures(config)
    dataset_version = dataset_fingerprint(config.get("input_path", ""))
    comparison_seed = config.get("evaluation_seed", 0)
    comparison_id = config.get("comparison_id") or hashlib.sha256(
        f"{generation_signature}:{comparison_seed}".encode("utf-8")).hexdigest()
    existing_ids = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    saved = json.loads(line)
                    saved_experiment = saved.get("experiment_id")
                    if saved_experiment != experiment_id:
                        raise ValueError(f"Output file contains a different experiment ({saved_experiment}); use a separate output path")
                    existing_ids.add(saved.get("sample_id"))
    prompt_path = config.get("prompt_templates_path", "configs/prompts/templates.yaml")
    prompt_path = Path(prompt_path)
    if not prompt_path.is_absolute():
        from emotion_steering.utils.paths import resolve_path
        prompt_path = resolve_path(prompt_path)
    prompt_resources = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    results = []
    try:
        from tqdm.auto import tqdm
        record_iterator = tqdm(records, desc="生成回复", unit="条")
    except ImportError:
        record_iterator = records
    for record in record_iterator:
        sample_id = record.get("sample_id", record.get("id"))
        if sample_id in existing_ids:
            continue
        system_prompt, applied = prepare_steering(model, record, config)
        target_language = config.get("target_language", record.get("language", "eng"))
        base_system = prompt_resources.get("languages", {}).get(target_language, {}).get("base")
        if not base_system:
            raise ValueError(f"No shared base instruction for target language {target_language!r}")
        sample_system = record.get("system", "")
        effective_system = "\n\n".join(part for part in (base_system, sample_system, system_prompt or "") if part)
        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": record["user"]})
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", **config.get("chat_template", {})).to(model.device)
        model._emotion_steering_generation_prompt_length = int(inputs.shape[-1])
        model._emotion_steering_generation_phase = "prefill"
        seed = int(config.get("generation_seed", config.get("evaluation_seed", 0)))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            generated = model.generate(inputs, **config.get("generation", {}))
        finally:
            model._emotion_steering_generation_prompt_length = None
        response = tokenizer.decode(generated[0, inputs.shape[-1]:], skip_special_tokens=True)
        hook_call_counts = getattr(getattr(model, "_emotion_steering_steer_fn", None), "call_counts", None)
        hook_observed_counts = getattr(getattr(model, "_emotion_steering_steer_fn", None), "observed_counts", None)
        forward_phase_counts = getattr(getattr(model, "_emotion_steering_steer_fn", None), "forward_phase_counts", None)
        applied_method = applied.get("method") if isinstance(applied, dict) else getattr(applied, "method", None)
        stage_not_executed = {}
        if applied_method in {"meandiff", "probe"}:
            required_phases = {"prefill", "generation"} if config.get("stage", "both") == "both" else {config.get("stage", "both")}
            stage_not_executed = validate_hook_execution(required_phases, applied["layers"],
                                                         hook_call_counts, forward_phase_counts)
        if hasattr(model, "set_steer_fn"):
            model.set_steer_fn(lambda activations, hook: activations)
        steering_metadata = (applied.metadata if hasattr(applied, "metadata") else applied)
        result = {"sample_id": sample_id,
                        "experiment_id": experiment_id, "generation_signature": generation_signature,
                        "comparison_id": comparison_id, "dataset_version": dataset_version,
                        "source_language": config.get("source_language"),
                        "target_language": target_language,
                        "target_emotion": config.get("target_emotion", config.get("emotion")),
                        "emotion": config.get("target_emotion", config.get("emotion")),
                        "method": config["method"], "layers": applied.get("layers") if isinstance(applied, dict) else None,
                        "coefficient": config.get("coefficient"), "user": record["user"],
                        "scenario_id": record.get("scenario_id"),
                        "system": record.get("system", ""),
                        "effective_system_prompt": effective_system,
                        "prompt_token_count": int(inputs.shape[-1]),
                        "steering_metadata": steering_metadata,
                        "response": response,
                        "hook_call_counts": hook_call_counts,
                        "hook_observed_counts": hook_observed_counts,
                        "forward_phase_counts": forward_phase_counts,
                        "stage_not_executed": stage_not_executed,
                        "generation_config": config.get("generation", {})}
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**run_metadata, **result}, ensure_ascii=False, allow_nan=False) + "\n")
        existing_ids.add(sample_id)
        results.append(result)
    return results


def run_from_config(config):
    """按配置加载模型并启动生成实验。"""
    records = [json.loads(line) for line in Path(config["input_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    target_language = config.get("target_language")
    if target_language:
        records = [row for row in records if row.get("language", target_language) == target_language]
    if config.get("parallel_evaluation", False):
        from emotion_steering.data.parallel import validate_parallel_records
        from emotion_steering.data.brighter import LANGUAGES
        validate_parallel_records(records, config.get("required_languages", LANGUAGES),
                                  require_reviewed=config.get("require_translation_review", True))
    model_name = config["model"]
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, hooked=config.get("method") in {"meandiff", "probe"},
                       device_map=config.get("device", "auto"),
                       torch_dtype=config.get("torch_dtype", "bfloat16"))
    if config.get("method") in {"sft", "dpo"}:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("Loading trained SFT/DPO adapters requires peft") from error
        if not config.get("adapter_path"):
            raise ValueError(f"{config['method'].upper()} inference requires config.adapter_path")
        model = PeftModel.from_pretrained(model, config["adapter_path"])
    model.eval()
    return run_records(model, tokenizer, records, config, config["output_path"])
