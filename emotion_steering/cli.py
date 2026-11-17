"""读取配置并分发项目各项命令。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

import yaml

from emotion_steering.data.brighter import LANGUAGES
from emotion_steering.methods.registry import METHODS
from emotion_steering.utils.paths import PROJECT_ROOT, resolve_path

LOG = logging.getLogger("emotion_steering")
PATH_KEYS = {"source_dir", "output_dir", "data_root", "report_dir", "input_path",
             "output_path", "vector_root", "vector_path", "data_path", "adapter_path", "base_config"}


def read_config(path):
    """读取 YAML 配置并解析其中的路径。"""
    path = resolve_path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    for key in PATH_KEYS.intersection(config):
        if config[key] is not None:
            config[key] = str(resolve_path(config[key]))
    return config


def validate_extraction_config(config):
    """检查向量提取配置是否有效。"""
    required = ("model", "languages", "emotions", "methods")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Extraction config is missing: {missing}")
    invalid_langs = sorted(set(config["languages"]) - set(LANGUAGES))
    if invalid_langs:
        raise ValueError(f"Unsupported languages: {invalid_langs}")
    invalid_methods = sorted(set(config["methods"]) - {"meandiff", "probe"})
    if invalid_methods:
        raise ValueError(f"Vector extraction supports meandiff/probe, got {invalid_methods}")
    return True


def dry_run(config_path=None):
    """检查配置和本地数据，不加载模型。"""
    report = {"python_config": str(config_path) if config_path else None,
              "registered_methods": sorted(METHODS), "languages": list(LANGUAGES),
              "processed_data_exists": resolve_path("data/processed/brighter/sampling_manifest.json").is_file(),
              "model_loaded": False, "checks": []}
    if config_path:
        config = read_config(config_path)
        if "methods" in config and "languages" in config:
            validate_extraction_config(config)
        manifest_path = resolve_path(config.get("data_root", "data/processed/brighter")) / "sampling_manifest.json"
        if manifest_path.is_file() and "languages" in config and "emotions" in config:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid, skipped = [], []
            for language in config["languages"]:
                for emotion in config["emotions"]:
                    task = manifest.get("languages", {}).get(language, {}).get("emotions", {}).get(emotion)
                    counts = task.get("counts", {}) if task else {}
                    if counts.get("0", {}).get("selected", 0) and counts.get("1", {}).get("selected", 0):
                        valid.append(f"{language}/{emotion}")
                    else:
                        skipped.append({"language": language, "emotion": emotion, "reason": "no_valid_binary_labels"})
            report["valid_tasks"] = valid
            report["skipped_tasks"] = skipped
        else:
            report["data_manifest"] = str(manifest_path)
        report["checks"].append("configuration valid")
    report["checks"].append("no model loaded; dry-run performs configuration and local-data checks only")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def resolve_run_all_config(config_value, project_root=None):
    """解析批量任务的配置文件路径。"""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    path = Path(config_value)
    return path if path.is_absolute() else root / path


def run_all(config_path):
    """按配置顺序执行多个实验命令。"""
    config = read_config(config_path)
    steps = config.get("steps", [])
    if not steps:
        raise ValueError("run-all needs nonempty steps, each with command and config fields")
    allowed = {"prepare-data", "validate-data", "extract-vectors", "train-peft", "analyze-vectors", "run-steering", "run-matrix", "evaluate"}
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("command") not in allowed or not step.get("config"):
            raise ValueError(f"Invalid run-all step {index}: expected command/config mapping")
        step_config = resolve_run_all_config(step["config"])
        LOG.info("Running step %d/%d: %s", index + 1, len(steps), step["command"])
        project_root = PROJECT_ROOT
        subprocess.run([sys.executable, str(project_root / "run.py"), step["command"], "--config", str(step_config)],
                       check=True, cwd=project_root)


def main():
    """解析命令行参数并运行对应任务。"""
    parser = argparse.ArgumentParser(description="Multilingual emotion steering experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-data", "validate-data", "extract-vectors", "train-peft", "analyze-vectors", "run-steering", "run-matrix", "evaluate", "run-all", "dry-run"):
        item = sub.add_parser(command)
        item.add_argument("--config", help="YAML configuration path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "dry-run":
        dry_run(args.config)
    elif args.command == "prepare-data":
        config = read_config(args.config) if args.config else {}
        from emotion_steering.data.preprocess import preprocess
        sampling = config.get("sampling", {})
        preprocess(resolve_path(config.get("source_dir", "data/raw/brighter")),
                   resolve_path(config.get("output_dir", "data/processed/brighter")),
                   per_class=sampling.get("per_class", 50), seed=sampling.get("seed", 42),
                   sampling_mode=sampling.get("mode", "balanced"))
    elif args.command == "validate-data":
        from emotion_steering.data.inspect import inspect
        config = read_config(args.config) if args.config else {}
        splits, labels = inspect(resolve_path(config.get("source_dir", "data/raw/brighter")),
                                 resolve_path(config.get("report_dir", "reports/brighter")))
        print(f"Validated {len(splits)} splits and {len(labels)} label distributions")
    elif args.command == "extract-vectors":
        config = read_config(args.config)
        validate_extraction_config(config)
        from emotion_steering.experiments.extract import run
        run(config)
    elif args.command == "analyze-vectors":
        from emotion_steering.analysis.vectors import run
        run(read_config(args.config))
    elif args.command == "evaluate":
        from emotion_steering.experiments.evaluate import evaluate_file
        config = read_config(args.config)
        output_path = config.get("output_path")
        if not output_path and config.get("output_dir"):
            output_path = str(Path(config["output_dir"]) / "evaluation.json")
        if not output_path:
            raise ValueError("Evaluation config requires output_path or output_dir")
        result = evaluate_file(config["input_path"], output_path, config.get("task", "classification"),
                               config.get("evaluator"), config.get("aggregation"))
        print(f"Saved evaluation result to {output_path}")
    elif args.command == "train-peft":
        from emotion_steering.methods.peft import run
        run(read_config(args.config))
    elif args.command == "run-steering":
        from emotion_steering.experiments.generate import run_from_config
        run_from_config(read_config(args.config))
    elif args.command == "run-matrix":
        from emotion_steering.experiments.experiment_runner import run_matrix
        results = run_matrix(read_config(args.config))
        print(f"Completed {len(results)} experiment conditions")
    elif args.command == "run-all":
        run_all(args.config)


if __name__ == "__main__":
    main()
