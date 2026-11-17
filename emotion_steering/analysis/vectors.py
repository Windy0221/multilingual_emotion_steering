"""运行向量统计与可视化分析。"""

from __future__ import annotations

from pathlib import Path

from emotion_steering.analysis.plots import plot_layer_similarity
from emotion_steering.analysis.plots import export_norm_probe_table
from emotion_steering.analysis.plots import project_vectors
from emotion_steering.analysis.plots import plot_similarity


def run(config):
    """根据配置执行模型训练或实验。"""
    model = config.get("model")
    if not model:
        raise ValueError("Analysis config must explicitly specify model")
    vector_root = Path(config.get("vector_root", "outputs/vectors"))
    output_dir = Path(config.get("output_dir", "outputs/analysis"))
    emotions = config.get("emotions", ["joy"])
    methods = config.get("methods", ["meandiff", "probe"])
    layers = config.get("layers", [0])
    reference = config.get("reference_language", "eng")
    results = []
    for emotion in emotions:
        for method in methods:
            for layer in layers:
                try:
                    plot_similarity(vector_root, emotion, method, int(layer), output_dir, model)
                    plot_layer_similarity(vector_root, emotion, method, reference, output_dir, model)
                    results.append({"emotion": emotion, "method": method, "layer": int(layer), "status": "ok"})
                except (ValueError, FileNotFoundError) as error:
                    results.append({"emotion": emotion, "method": method, "layer": int(layer), "status": "skipped", "reason": str(error)})
            try:
                project_vectors(vector_root, method, int(layers[0]), output_dir, model)
            except (ValueError, FileNotFoundError) as error:
                results.append({"emotion": "all", "method": method, "status": "skipped_projection", "reason": str(error)})
    export_norm_probe_table(vector_root, output_dir, model=model)
    return results
