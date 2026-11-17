"""收集向量产物并生成分析图表。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from emotion_steering.utils.io import load_vector_artifact


EXTRACTION_SIGNATURE_FIELDS = ("token_mode", "pooling", "hook_locations", "hook_name_template",
                               "normalization", "vector_dim", "hidden_size", "chat_template")


def _iter_artifacts(root, method, model, emotion=None):
    """遍历向量产物目录。"""
    if not model:
        raise ValueError("Vector analysis requires an explicit model identifier")
    root = Path(root)
    pattern = f"**/{method}/*/{emotion}/metadata.json" if emotion else f"**/{method}/*/*/metadata.json"
    signature = None
    seen_languages = set()
    for metadata_path in sorted(root.glob(pattern)):
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("model") != model:
            continue
        missing_fields = [field for field in EXTRACTION_SIGNATURE_FIELDS if field not in meta]
        if missing_fields:
            raise ValueError(f"Vector metadata for model={model} is missing extraction settings: {missing_fields}")
        current = tuple(tuple(meta[field]) if field == "hook_locations" else meta[field]
                        for field in EXTRACTION_SIGNATURE_FIELDS)
        if signature is None:
            signature = current
        elif current != signature:
            raise ValueError(f"Cannot compare vectors with different extraction settings for model={model}")
        language = meta.get("source_language")
        if emotion and language in seen_languages:
            raise ValueError(f"Multiple vector artifacts found for model={model}, language={language}, emotion={emotion}; refusing to overwrite")
        if emotion:
            seen_languages.add(language)
        yield metadata_path, meta


def collect_vectors(root, emotion, method, layer, model):
    """读取并整理待分析的向量。"""
    items = {}
    vector_dim = None
    for metadata_path, meta in _iter_artifacts(root, method, model, emotion):
        artifact_dir = metadata_path.parent
        language = meta.get("source_language")
        vectors, _ = load_vector_artifact(artifact_dir)
        layer_vector = vectors.get(int(layer))
        if layer_vector is not None and torch.isfinite(layer_vector).all() and torch.linalg.vector_norm(layer_vector) > 0:
            current_dim = int(layer_vector.numel())
            if vector_dim is not None and current_dim != vector_dim:
                raise ValueError(f"Cannot compare vector dimensions {vector_dim} and {current_dim} for model={model}")
            vector_dim = current_dim
            items[language] = layer_vector.flatten().numpy()
    return items


def plot_similarity(root, emotion, method, layer, output_dir, model):
    """绘制向量相似度热图。"""
    items = collect_vectors(root, emotion, method, layer, model)
    languages = sorted(items)
    if not languages:
        raise ValueError(f"No valid vector artifacts for emotion={emotion}, method={method}, layer={layer}")
    matrix = np.full((len(languages), len(languages)), np.nan)
    for i, left in enumerate(languages):
        for j, right in enumerate(languages):
            a, b = items[left], items[right]
            if a.shape == b.shape:
                matrix[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"similarity_{emotion}_{method}_layer{layer}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream); writer.writerow(["language", *languages])
        for language, row in zip(languages, matrix): writer.writerow([language, *row.tolist()])
    fig, ax = plt.subplots(figsize=(max(5, len(languages) * .6), max(4, len(languages) * .55)))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(languages)), languages, rotation=45, ha="right"); ax.set_yticks(range(len(languages)), languages)
    ax.set_title(f"{emotion} {method} cosine similarity, layer {layer}")
    fig.colorbar(image, ax=ax, label="cosine similarity"); fig.tight_layout()
    fig.savefig(out / f"similarity_{emotion}_{method}_layer{layer}.png", dpi=180); plt.close(fig)
    return matrix







def plot_layer_similarity(root, emotion, method, reference, output_dir, model):
    """绘制逐层向量相似度变化图。"""
    root = Path(root); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    by_language = {}
    vector_dim = None
    for metadata_path, metadata in _iter_artifacts(root, method, model, emotion):
        directory = metadata_path.parent
        vectors, _ = load_vector_artifact(directory)
        dimensions = {int(vector.numel()) for vector in vectors.values()}
        if len(dimensions) > 1:
            raise ValueError(f"Vector layers have inconsistent dimensions for {metadata_path}")
        current_dim = next(iter(dimensions), None)
        if vector_dim is not None and current_dim is not None and current_dim != vector_dim:
            raise ValueError(f"Cannot compare vector dimensions {vector_dim} and {current_dim} for model={model}")
        if current_dim is not None:
            vector_dim = current_dim
        by_language[metadata["source_language"]] = vectors
    if not by_language:
        raise ValueError(f"No valid vector artifacts for emotion={emotion}, method={method}")
    rows = []
    for language, vectors in sorted(by_language.items()):
        for layer, vector in vectors.items():
            ref = by_language.get(reference, {}).get(layer)
            score = float("nan")
            if ref is not None and ref.shape == vector.shape and torch.linalg.vector_norm(ref) and torch.linalg.vector_norm(vector):
                score = float(torch.nn.functional.cosine_similarity(ref.flatten(), vector.flatten(), dim=0))
            rows.append((language, int(layer), score))
    csv_path = output / f"layer_similarity_{emotion}_{method}_{reference}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream); writer.writerow(["language", "layer", "cosine_similarity"]); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for language in sorted(by_language):
        subset = [(layer, score) for lang, layer, score in rows if lang == language]
        ax.plot([x[0] for x in subset], [x[1] for x in subset], marker=".", label=language)
    ax.set(xlabel="Layer", ylabel="Cosine similarity", ylim=(-1, 1), title=f"{emotion} {method} vs {reference}")
    ax.legend(ncol=2); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / f"layer_similarity_{emotion}_{method}_{reference}.png", dpi=180); plt.close(fig)
    return rows







def project_vectors(root, method, layer, output_dir, model):
    """将向量投影到二维空间。"""
    root = Path(root); rows = []
    for metadata_path, metadata in _iter_artifacts(root, method, model):
        vectors, _ = load_vector_artifact(metadata_path.parent)
        vector = vectors.get(int(layer))
        if vector is not None and torch.isfinite(vector).all() and torch.linalg.vector_norm(vector) > 0:
            rows.append((metadata["source_language"], metadata["emotion"], vector.flatten().numpy()))
    if not rows: raise ValueError("PCA projection has no finite, nonzero vector artifacts")
    dims = len(rows[0][2])
    if any(len(row[2]) != dims for row in rows):
        raise ValueError(f"Cannot project vectors with mixed feature dimensions for model={model}")
    if dims < 2:
        raise ValueError(f"PCA projection needs vectors with at least two features; found dimension {dims}")
    if len(rows) < 3:
        raise ValueError(f"PCA projection needs at least three compatible vectors after dimension filtering; found {len(rows)}")
    coordinates = PCA(n_components=2).fit_transform(np.stack([row[2] for row in rows]))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / f"projection_{method}_layer{layer}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream); writer.writerow(["language", "emotion", "pc1", "pc2"])
        writer.writerows([[row[0], row[1], *point.tolist()] for row, point in zip(rows, coordinates)])
    fig, ax = plt.subplots(figsize=(8, 6))
    for emotion in sorted({row[1] for row in rows}):
        idx = [i for i, row in enumerate(rows) if row[1] == emotion]
        ax.scatter(coordinates[idx, 0], coordinates[idx, 1], label=emotion)
        for i in idx: ax.annotate(rows[i][0], coordinates[i], fontsize=7, alpha=.8)
    ax.set_title(f"Joint PCA of {method} vectors, layer {layer}"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out / f"projection_{method}_layer{layer}.png", dpi=180); plt.close(fig)
    return coordinates







def export_norm_probe_table(root, output_dir, model=None):
    """导出向量范数和 Probe 指标表。"""
    rows = []
    for metadata_path in Path(root).glob("**/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if model and metadata.get("model") != model:
            continue
        vectors, _ = load_vector_artifact(metadata_path.parent)
        metrics = metadata.get("probe_metrics", {})
        for layer, vector in vectors.items():
            metric = metrics.get(str(layer), {})
            rows.append({"model": metadata.get("model"), "language": metadata.get("source_language"),
                         "emotion": metadata.get("emotion"), "method": metadata.get("method"),
                         "layer": int(layer), "norm": float(torch.linalg.vector_norm(vector)),
                         "accuracy": metric.get("accuracy"), "macro_f1": metric.get("macro_f1")})
    if not rows:
        raise ValueError(f"No vector artifacts found under {root}")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "vector_norm_probe_metrics.csv"
    fields = ["model", "language", "emotion", "method", "layer", "norm", "accuracy", "macro_f1"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    finite = [row for row in rows if row["macro_f1"] is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in finite if row["method"] == method]
        ax.scatter([row["norm"] for row in subset], [row["macro_f1"] for row in subset], label=method, alpha=.7)
    ax.set(xlabel="Vector norm", ylabel="Validation macro F1", title="Vector norm and probe performance")
    if finite: ax.legend()
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output / "vector_norm_probe_metrics.png", dpi=180); plt.close(fig)
    return rows
