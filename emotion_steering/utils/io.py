"""读取 JSONL 并安全保存、加载向量产物。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件并返回记录列表。"""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    return records


def save_vector_artifact(path: str | Path, vectors: dict[int, torch.Tensor], metadata: dict[str, Any],
                         extra_files: dict[str, Any] | None = None) -> None:
    """将向量和元数据安全写入产物目录。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            load_vector_artifact(target, {key: value for key, value in metadata.items() if key not in {"probe_classifier_artifact"}})
            return
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
            raise ValueError(f"Existing vector artifact at {target} is incomplete or incompatible; remove it before retrying") from error
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        _write_vector_artifact(staging, vectors, metadata)
        for name, content in (extra_files or {}).items():
            torch.save(content, staging / name)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_vector_artifact(target: Path, vectors: dict[int, torch.Tensor], metadata: dict[str, Any]) -> None:
    """把向量文件及元数据写入临时目录。"""
    cpu_vectors = {int(layer): torch.as_tensor(vector).detach().float().cpu() for layer, vector in vectors.items()}
    torch.save(cpu_vectors, target / "vectors.pt")
    enriched = dict(metadata)
    enriched["layers"] = sorted(cpu_vectors)
    enriched["vector_dim"] = int(next(iter(cpu_vectors.values())).numel()) if cpu_vectors else None
    enriched["vector_norms"] = {str(layer): float(torch.linalg.vector_norm(vector)) for layer, vector in cpu_vectors.items()}
    (target / "metadata.json").write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vector_artifact(path: str | Path, expected: dict[str, Any] | None = None):
    """读取向量产物并检查元数据。"""
    target = Path(path)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"Vector metadata mismatch for {key}: expected {value!r}, found {metadata.get(key)!r}")
    vectors = torch.load(target / "vectors.pt", map_location="cpu", weights_only=True)
    classifier_file = metadata.get("probe_classifier_artifact")
    if classifier_file and not (target / classifier_file).is_file():
        raise ValueError(f"Vector artifact is incomplete: missing {classifier_file}")
    if set(map(int, vectors)) != set(metadata.get("layers", [])):
        raise ValueError("Vector layer keys do not match metadata")
    return vectors, metadata


def file_sha256(path: str | Path) -> str:
    """计算文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
