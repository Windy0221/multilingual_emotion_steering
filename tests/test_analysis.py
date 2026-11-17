"""测试文件：检查项目功能是否符合预期。"""

import torch
import pytest

from emotion_steering.utils.io import save_vector_artifact
from emotion_steering.analysis.plots import plot_layer_similarity
from emotion_steering.analysis.plots import export_norm_probe_table
from emotion_steering.analysis.plots import project_vectors
from emotion_steering.analysis.plots import plot_similarity


EXTRACTION_SETTINGS = {"token_mode": "user_all", "pooling": "mean", "hook_locations": [7],
                      "hook_name_template": "model.layers.{layer}.hook_after_mlp_hs",
                      "normalization": "none", "chat_template": {"enable_thinking": True}}


def test_vector_plots_handle_saved_artifacts(tmp_path):
    """验证vector plots handle saved artifacts行为。"""
    root = tmp_path / "vectors"
    for lang, emotion, vec in [
        ("eng", "joy", [1., 0., 1.]),
        ("chn", "joy", [0., 1., 1.]),
        ("eng", "anger", [-1., 0., 0.]),
    ]:
        target = root / "toy" / "meandiff" / lang / emotion
        save_vector_artifact(target, {0: torch.tensor(vec)}, {
            "model": "toy", "source_language": lang, "emotion": emotion,
            "method": "meandiff", "hidden_size": len(vec),
            "probe_metrics": {"0": {"macro_f1": 0.7, "accuracy": 0.8}},
            **EXTRACTION_SETTINGS,
        })
    output = tmp_path / "plots"
    matrix = plot_similarity(root, "joy", "meandiff", 0, output, "toy")
    assert matrix.shape == (2, 2)
    assert (output / "similarity_joy_meandiff_layer0.csv").exists()
    assert len(plot_layer_similarity(root, "joy", "meandiff", "eng", output, "toy")) == 2
    assert project_vectors(root, "meandiff", 0, output, "toy").shape == (3, 2)
    assert len(export_norm_probe_table(root, output, model="toy")) == 3


def test_vector_plots_report_missing_artifacts_instead_of_blank_figures(tmp_path):
    """验证vector plots report missing artifacts instead of blank figures行为。"""
    with pytest.raises(ValueError, match="No valid vector artifacts"):
        plot_similarity(tmp_path, "joy", "meandiff", 0, tmp_path / "plots", "toy")
    with pytest.raises(ValueError, match="No valid vector artifacts"):
        plot_layer_similarity(tmp_path, "joy", "meandiff", "eng", tmp_path / "plots", "toy")
    with pytest.raises(ValueError, match="No vector artifacts found"):
        export_norm_probe_table(tmp_path, tmp_path / "plots")


def test_pca_rejects_mixed_dimensions(tmp_path):
    """验证pca rejects mixed dimensions行为。"""
    root = tmp_path / "vectors"
    records = [("eng", [1., 0., 0.]), ("chn", [0., 1., 0.]), ("esp", [1., 1.])]
    for lang, vector in records:
        target = root / "toy" / "meandiff" / lang / "joy"
        save_vector_artifact(target, {0: torch.tensor(vector)}, {
            "model": "toy", "source_language": lang, "emotion": "joy", "method": "meandiff",
            "hidden_size": len(vector),
            **EXTRACTION_SETTINGS,
        })
    with pytest.raises(ValueError, match="different extraction settings"):
        project_vectors(root, "meandiff", 0, tmp_path / "plots", "toy")


def test_vector_analysis_filters_model_and_rejects_mixed_extraction_settings(tmp_path):
    """验证vector analysis filters model and rejects mixed extraction settings行为。"""
    root = tmp_path / "vectors"
    for model, lang, settings in [
        ("toy-a", "eng", EXTRACTION_SETTINGS),
        ("toy-a", "chn", EXTRACTION_SETTINGS),
        ("toy-b", "esp", EXTRACTION_SETTINGS),
    ]:
        target = root / model / "meandiff" / lang / "joy"
        save_vector_artifact(target, {0: torch.tensor([1., 0.])}, {
            "model": model, "source_language": lang, "emotion": "joy", "method": "meandiff",
            "hidden_size": 2, **settings,
        })
    matrix = plot_similarity(root, "joy", "meandiff", 0, tmp_path / "plots", "toy-a")
    assert matrix.shape == (2, 2)

    mixed = root / "toy-a" / "meandiff" / "deu" / "joy"
    save_vector_artifact(mixed, {0: torch.tensor([0., 1.])}, {
        "model": "toy-a", "source_language": "deu", "emotion": "joy", "method": "meandiff",
        "hidden_size": 2,
        **{**EXTRACTION_SETTINGS, "token_mode": "user_last", "pooling": "last"},
    })
    with pytest.raises(ValueError, match="different extraction settings"):
        plot_similarity(root, "joy", "meandiff", 0, tmp_path / "plots", "toy-a")


def test_vector_analysis_requires_model_identifier(tmp_path):
    """验证vector analysis requires model identifier行为。"""
    with pytest.raises(ValueError, match="explicit model"):
        plot_similarity(tmp_path, "joy", "meandiff", 0, tmp_path / "plots", "")
