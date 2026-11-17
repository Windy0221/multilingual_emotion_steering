"""通过注册表加载指定数据集。"""

from __future__ import annotations

from .brighter import LANGUAGES, load_brighter_local


DATASETS = {"brighter": load_brighter_local}


def load_dataset(name: str, **kwargs):
    """通过注册名加载数据集。"""
    try:
        loader = DATASETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset {name!r}; available datasets: {', '.join(DATASETS)}") from error
    return loader(**kwargs)
