"""定义 Steering 方法的通用接口和产物结构。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MethodArtifact:
    method: str
    artifact_type: str
    payload: Any
    metadata: dict[str, Any]


class SteeringMethod(ABC):
    name = "base"
    artifact_type = "unknown"

    def prepare(self, data: Any, config: dict[str, Any]) -> Any:
        """执行当前测试或辅助逻辑。"""
        return data

    @abstractmethod
    def fit(self, model: Any, data: Any, config: dict[str, Any]) -> MethodArtifact:
        """根据数据拟合当前 Steering 方法。"""
        raise NotImplementedError

    @abstractmethod
    def apply(self, model: Any, artifact: MethodArtifact, config: dict[str, Any]) -> Any:
        """将当前方法应用到模型或生成配置。"""
        raise NotImplementedError

    def save(self, artifact: MethodArtifact, path: str) -> None:
        """保存方法产物及元数据。"""
        raise NotImplementedError(f"{self.name} does not implement artifact saving")

    def load(self, path: str) -> MethodArtifact:
        """读取方法产物。"""
        raise NotImplementedError(f"{self.name} does not implement artifact loading")
