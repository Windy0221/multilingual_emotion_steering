"""根据隐藏状态计算 MeanDiff 和 Probe 向量。"""

from __future__ import annotations

import torch

from .base import MethodArtifact, SteeringMethod


class MeanDiffSteering(SteeringMethod):
    name = "meandiff"
    artifact_type = "activation_vector"

    def fit(self, model, data, config):
        """根据数据拟合当前 Steering 方法。"""
        positive = torch.as_tensor(data["positive"], dtype=torch.float32)
        negative = torch.as_tensor(data["negative"], dtype=torch.float32)
        if positive.ndim < 2 or negative.ndim != positive.ndim or positive.shape[1:] != negative.shape[1:]:
            raise ValueError("Positive and negative activations must have matching [N, ...] shapes")
        if len(positive) == 0 or len(negative) == 0:
            raise ValueError("MeanDiff needs at least one positive and negative activation")
        vector = positive.mean(dim=0) - negative.mean(dim=0)
        return MethodArtifact(self.name, self.artifact_type, vector, {"positive_count": len(positive), "negative_count": len(negative), "pooling": config.get("pooling", "provided")})

    def apply(self, model, artifact, config):
        """将当前方法应用到模型或生成配置。"""
        if artifact.artifact_type != self.artifact_type:
            raise TypeError("MeanDiff requires an activation_vector artifact")
        return artifact.payload



import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .base import MethodArtifact, SteeringMethod


class ProbeSteering(SteeringMethod):
    name = "probe"
    artifact_type = "probe_and_vector"

    def fit(self, model, data, config):
        """根据数据拟合当前 Steering 方法。"""
        positive = torch.as_tensor(data["positive"], dtype=torch.float32)
        negative = torch.as_tensor(data["negative"], dtype=torch.float32)
        if len(positive) == 0 or len(negative) == 0 or positive.shape[1:] != negative.shape[1:]:
            raise ValueError("Probe needs nonempty positive/negative activations with matching shapes")
        x = torch.cat([positive, negative]).reshape(len(positive) + len(negative), -1).cpu().numpy()
        y = np.concatenate([np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)])
        classifier = LogisticRegression(random_state=int(config.get("seed", 42)), max_iter=int(config.get("max_iter", 1000)))
        classifier.fit(x, y)
        vector = torch.as_tensor(classifier.coef_[0], dtype=torch.float32).reshape(positive.shape[1:])
        return MethodArtifact(self.name, self.artifact_type, {"vector": vector, "classifier": classifier}, {"positive_count": len(positive), "negative_count": len(negative), "train_accuracy": float(classifier.score(x, y))})

    def apply(self, model, artifact, config):
        """将当前方法应用到模型或生成配置。"""
        if artifact.artifact_type != self.artifact_type:
            raise TypeError("Probe requires a probe_and_vector artifact")
        return artifact.payload["vector"]
