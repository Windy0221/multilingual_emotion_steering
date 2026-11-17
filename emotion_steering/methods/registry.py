"""注册并创建可用的 Steering 方法。"""

from __future__ import annotations

from .peft import DPOSteering, SFTSteering
from .vector import MeanDiffSteering, ProbeSteering
from .prompt import DescriptiveSteering, FewShotSteering, NoSteering, ZeroShotSteering


METHODS = {
    "none": NoSteering,
    "zero_shot": ZeroShotSteering,
    "few_shot": FewShotSteering,
    "descriptive": DescriptiveSteering,
    "meandiff": MeanDiffSteering,
    "probe": ProbeSteering,
    "sft": SFTSteering,
    "dpo": DPOSteering,
}


def create_method(name: str):
    """根据注册名创建 Steering 方法。"""
    try:
        return METHODS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown steering method {name!r}; available: {', '.join(METHODS)}") from error
