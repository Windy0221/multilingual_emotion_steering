"""JSONL adapter for multilingual dialogue and evaluation records."""
from emotion_steering.utils.io import load_jsonl


def load_dialogue_jsonl(path):
    """读取并整理 JSONL 对话数据。"""
    rows = load_jsonl(path)
    for index, row in enumerate(rows):
        if not row.get("sample_id") or not row.get("language") or not isinstance(row.get("user"), str) or not row["user"].strip():
            raise ValueError(f"Dialogue record {index} needs sample_id, language, and nonempty user")
        if row.get("reference") is not None and not isinstance(row["reference"], str):
            raise ValueError(f"Dialogue record {index} reference must be text or null")
    return rows
