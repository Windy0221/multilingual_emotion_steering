"""测试文件：检查项目功能是否符合预期。"""

from emotion_steering.cli import read_config
from emotion_steering.experiments.experiment_runner import expand_matrix
from emotion_steering.utils.paths import PROJECT_ROOT, resolve_path


def test_extraction_and_analysis_configs_resolve_from_project_root():
    """验证extraction and analysis configs resolve from project root行为。"""
    extraction = read_config("configs/extraction.yaml")
    analysis = read_config("configs/analysis.yaml")
    assert extraction["data_root"] == str(resolve_path("data/processed/brighter"))
    assert extraction["output_dir"] == str(resolve_path("outputs/vectors"))
    assert extraction["chat_template"] == {"enable_thinking": False}
    assert extraction["tokenization"] == {"add_special_tokens": False}
    steering = read_config("configs/steering.yaml")
    assert steering["chat_template"] == extraction["chat_template"]
    assert analysis["model"] == "Qwen/Qwen3-8B"
    assert analysis["vector_root"] == str(resolve_path("outputs/vectors/Qwen_Qwen3-8B/qwen3_8b_seed42"))
    assert PROJECT_ROOT.name == "multilingual_emotion_steering"


def test_matrix_includes_shared_baselines_and_source_language_conditions():
    """验证实验矩阵为每个目标条件加入基线和来源语言组合。"""
    conditions = expand_matrix({"target_languages": ["chn"], "target_emotions": ["joy"],
                                "methods": ["meandiff"], "vector_source_languages": ["eng", "chn"],
                                "coefficients": [0.5, 1.0], "layers": [[8], [16]]})
    assert len(conditions) == 9
    assert sum(item["method"] == "none" for item in conditions) == 1
    assert sum(item["method"] == "meandiff" for item in conditions) == 8
