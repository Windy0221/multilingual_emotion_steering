"""从 BRIGHTER 数据提取并保存情绪向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from emotion_steering.data.brighter import LANGUAGES, load_brighter_local
from emotion_steering.utils.io import file_sha256, load_vector_artifact, save_vector_artifact
from emotion_steering.models.loader import load_model, load_tokenizer
from emotion_steering.methods.vector import MeanDiffSteering
from emotion_steering.methods.vector import ProbeSteering
from emotion_steering.models.activations import (HOOK_LOCATION, HOOK_NAME_TEMPLATE,
                                                 extract_user_token_activations)
from emotion_steering.utils.paths import resolve_path

logger = logging.getLogger("emotion_steering.extract")


def _hash_ids(records):
    """计算样本 ID 列表的摘要。"""
    joined = "\n".join(str(row["id"]) for row in records).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def extract_task_activations(model, tokenizer, language, emotion, config):
    """提取指定语言和情绪任务的隐藏状态。"""
    processed = resolve_path(config.get("data_root", "data/processed/brighter"))
    sample_path = processed / "binary_train" / language / f"{emotion}.parquet"
    if not sample_path.exists():
        logger.warning("Skipping %s/%s: no sampled task file", language, emotion)
        return None
    frame = pd.read_parquet(sample_path)
    classes = set(frame["label"].astype(int).tolist()) if not frame.empty else set()
    if classes != {0, 1}:
        logger.warning("Skipping %s/%s: no_valid_binary_labels (classes=%s)", language, emotion, sorted(classes))
        return None
    token_mode = config.get("token_mode", "user_all")
    pooling = "last" if token_mode == "user_last" else config.get("pooling", "mean")
    if token_mode not in {"user_all", "user_last"}:
        raise ValueError("BRIGHTER input extraction supports user_all and user_last")
    activations = extract_user_token_activations(frame.to_dict("records"), tokenizer, model,
                                                batch_size=int(config.get("batch_size", 4)), pooling=pooling,
                                                chat_template_kwargs=config.get("chat_template", {}))
    return frame, activations


def extract_for_task(model, tokenizer, language, emotion, method, output_root, config, prepared=None):
    """计算并保存指定方法的情绪向量。"""
    prepared = prepared if prepared is not None else extract_task_activations(model, tokenizer, language, emotion, config)
    if prepared is None:
        return False
    frame, activations = prepared
    token_mode = config.get("token_mode", "user_all")
    pooling = "last" if token_mode == "user_last" else config.get("pooling", "mean")
    labels = frame["label"].to_numpy(dtype=int)
    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)), test_size=float(config.get("validation_fraction", 0.2)),
        random_state=int(config.get("seed", 42)), stratify=labels,
    )
    vectors: dict[int, torch.Tensor] = {}
    probe_parameters: dict[int, dict[str, torch.Tensor]] = {}
    metrics = {}
    for layer, states in activations.items():
        positives = states[train_indices][torch.as_tensor(labels[train_indices] == 1)]
        negatives = states[train_indices][torch.as_tensor(labels[train_indices] == 0)]
        if method == "meandiff":
            artifact = MeanDiffSteering().fit(None, {"positive": positives, "negative": negatives}, config)
            vectors[layer] = artifact.payload
        elif method == "probe":
            artifact = ProbeSteering().fit(None, {"positive": positives, "negative": negatives}, config)
            classifier = artifact.payload["classifier"]
            vectors[layer] = artifact.payload["vector"]
            probe_parameters[layer] = {
                "coef": torch.as_tensor(classifier.coef_[0], dtype=torch.float32),
                "intercept": torch.as_tensor(classifier.intercept_[0], dtype=torch.float32),
                "classes": torch.as_tensor(classifier.classes_, dtype=torch.int64),
            }
            predictions = classifier.predict(states[validation_indices].numpy())
            from sklearn.metrics import f1_score
            metrics[str(layer)] = {"accuracy": float(np.mean(predictions == labels[validation_indices])),
                                   "macro_f1": float(f1_score(labels[validation_indices], predictions, average="macro"))}
        else:
            raise ValueError("method must be meandiff or probe")
    model_name = str(config["model"])
    run_id = config.get("run_id", "default")
    target = Path(output_root) / model_name.replace("/", "_") / run_id / method / language / emotion
    sample_path = resolve_path(config.get("data_root", "data/processed/brighter")) / "binary_train" / language / f"{emotion}.parquet"
    dataset_sha256 = file_sha256(sample_path) if sample_path.is_file() else hashlib.sha256(
        frame.to_json(orient="records", force_ascii=False).encode("utf-8")).hexdigest()
    metadata = {
        "model": model_name, "source_language": language, "emotion": emotion,
        "method": method, "run_id": run_id, "token_mode": token_mode, "pooling": pooling,
        "hook": "transformer_block_output", "hook_locations": [HOOK_LOCATION],
        "hook_name_template": HOOK_NAME_TEMPLATE, "normalization": "none",
        "chat_template": config.get("chat_template", {}),
        "positive_count": int((labels[train_indices] == 1).sum()),
        "negative_count": int((labels[train_indices] == 0).sum()),
        "validation_count": int(len(validation_indices)), "probe_metrics": metrics,
        "seed": int(config.get("seed", 42)), "sample_ids_sha256": _hash_ids(frame.to_dict("records")),
        "dataset_sha256": dataset_sha256,
        "model_hidden_layers": len(activations),
        "hidden_size": int(next(iter(activations.values())).shape[-1]),
        "probe_classifier_artifact": "probe.pt" if method == "probe" else None,
    }
    save_vector_artifact(target, vectors, metadata,
                         extra_files={"probe.pt": probe_parameters} if probe_parameters else None)
    return True


def run(config):
    """根据配置执行模型训练或实验。"""
    model_name = config["model"]
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, hooked=True, device_map=config.get("device", "auto"),
                       torch_dtype=config.get("torch_dtype", "bfloat16"))
    successes = 0
    languages = config.get("languages", LANGUAGES)
    try:
        from tqdm.auto import tqdm
        language_iterator = tqdm(languages, desc="向量提取语言", unit="语言")
    except ImportError:
        language_iterator = languages
    for language in language_iterator:
        # Loading here validates availability; task extraction uses fixed sampled rows.
        try:
            _, _, concepts, _ = load_brighter_local(language, emotions=config.get("emotions"),
                                                    data_root=config.get("data_root"))
        except (FileNotFoundError, ValueError) as error:
            logger.warning("Skipping language %s: %s", language, error)
            continue
        try:
            from tqdm.auto import tqdm
            emotion_iterator = tqdm(concepts, desc=f"{language} 情绪任务", leave=False, unit="任务")
        except ImportError:
            emotion_iterator = concepts
        for emotion in emotion_iterator:
            methods = config.get("methods", ["meandiff", "probe"])
            output_root = config.get("output_dir", "outputs/vectors")
            model_dir = str(config["model"]).replace("/", "_")
            run_id = config.get("run_id", "default")
            pending = []
            sample_path = resolve_path(config.get("data_root", "data/processed/brighter")) / "binary_train" / language / f"{emotion}.parquet"
            dataset_sha256 = file_sha256(sample_path) if sample_path.is_file() else None
            for method in methods:
                target = Path(output_root) / model_dir / run_id / method / language / emotion
                if target.exists():
                    load_vector_artifact(target, {"model": config["model"], "source_language": language,
                                                  "emotion": emotion, "method": method, "run_id": run_id,
                                                  "chat_template": config.get("chat_template", {}),
                                                  "token_mode": config.get("token_mode", "user_all"),
                                                  "pooling": ("last" if config.get("token_mode") == "user_last"
                                                              else config.get("pooling", "mean")),
                                                  "seed": int(config.get("seed", 42)),
                                                  "hook_name_template": HOOK_NAME_TEMPLATE,
                                                  "dataset_sha256": dataset_sha256})
                    logger.info("Resuming: valid %s/%s/%s artifact already exists", method, language, emotion)
                else:
                    pending.append(method)
            if not pending:
                continue
            prepared = extract_task_activations(model, tokenizer, language, emotion, config)
            if prepared is None:
                continue
            for method in pending:
                successes += int(extract_for_task(model, tokenizer, language, emotion, method,
                                                   output_root, config,
                                                   prepared=prepared))
    logger.info("Saved %d vector artifacts", successes)
    return successes


def main():
    """解析命令行参数并运行对应任务。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    import yaml
    logging.basicConfig(level=logging.INFO)
    run(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
