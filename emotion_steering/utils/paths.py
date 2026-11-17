"""统一解析项目内路径。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, *, root: str | Path | None = None) -> Path:
    """将路径解析为项目内的绝对路径。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(root) if root is not None else PROJECT_ROOT) / path


def resolve_config_path(value: str | Path) -> Path:
    """解析配置文件路径。"""
    return resolve_path(value)
