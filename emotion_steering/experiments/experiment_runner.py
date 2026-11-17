"""根据实验矩阵展开独立 Steering 条件并依次运行。"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import yaml

from emotion_steering.utils.paths import resolve_path


VECTOR_METHODS = {"meandiff", "probe"}
PROMPT_METHODS = {"zero_shot", "few_shot", "descriptive"}


def _slug(value):
    """把条件值转换为适合文件名的短字符串。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "none"


def expand_matrix(config):
    """展开方法、目标语言、目标情绪和向量来源条件。"""
    methods = list(config.get("methods", []))
    if "none" not in methods:
        methods.insert(0, "none")
    vector_sources = config.get("vector_source_languages", [])
    coefficients = config.get("coefficients", [1.0])
    layers = config.get("layers", [[16]])
    if layers and isinstance(layers[0], int):
        layers = [layers]
    conditions = []
    for language, emotion, method in itertools.product(config["target_languages"], config["target_emotions"], methods):
        if method in VECTOR_METHODS:
            for source, coefficient, selected_layers in itertools.product(vector_sources, coefficients, layers):
                conditions.append({"target_language": language, "target_emotion": emotion,
                                   "method": method, "source_language": source,
                                   "coefficient": coefficient, "layers": selected_layers})
        elif method in PROMPT_METHODS:
            conditions.append({"target_language": language, "target_emotion": emotion,
                               "method": method, "source_language": None,
                               "coefficient": None, "layers": None})
        elif method in {"none", "sft", "dpo"}:
            conditions.append({"target_language": language, "target_emotion": emotion,
                               "method": method, "source_language": None,
                               "coefficient": None, "layers": None})
        else:
            raise ValueError(f"Unsupported method in experiment matrix: {method}")
    return conditions


def run_matrix(config):
    """运行矩阵中的每个实验条件并保存独立结果文件。"""
    conditions = expand_matrix(config)
    base_config_path = resolve_path(config["base_config"])
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    output_dir = resolve_path(config.get("output_dir", "outputs/generations/matrix"))
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = resolve_path(config.get("input_path", base["input_path"]))
    total = len(conditions)
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(enumerate(conditions, 1), total=total, desc="实验条件")
    except ImportError:
        iterator = enumerate(conditions, 1)
    results = []
    for index, condition in iterator:
        run_config = {**base, **condition, "input_path": str(input_path),
                      "comparison_id": config.get("comparison_id", "matrix-default"),
                      "evaluation_seed": config.get("evaluation_seed", 0),
                      "generation": {**base.get("generation", {}), **config.get("generation", {})}}
        if condition["method"] in VECTOR_METHODS:
            vector_root = resolve_path(config.get("vector_root", "outputs/vectors"))
            run_id = config.get("vector_run_id", "default")
            run_config["vector_path"] = str(vector_root / str(run_config["model"]).replace("/", "_") /
                                             run_id / condition["method"] / condition["source_language"] /
                                             condition["target_emotion"])
            run_config["normalize"] = config.get("normalize", run_config.get("normalize", False))
        name_parts = [condition["target_language"], condition["target_emotion"], condition["method"]]
        name_parts.extend(_slug(condition[key]) for key in ("source_language", "coefficient") if condition.get(key) is not None)
        if condition.get("layers") is not None:
            name_parts.append("L" + "-".join(map(str, condition["layers"])))
        run_config["output_path"] = str(output_dir / ("_".join(name_parts) + ".jsonl"))
        from emotion_steering.experiments.generate import run_from_config
        results.append({"condition": condition, "output_path": run_config["output_path"],
                        "generated": len(run_from_config(run_config))})
    return results
