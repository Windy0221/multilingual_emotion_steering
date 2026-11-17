"""测试文件：检查项目功能是否符合预期。"""

import json
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from emotion_steering.models.activations import extract_user_token_activations
from emotion_steering.utils.io import load_vector_artifact, save_vector_artifact
from emotion_steering.experiments.evaluate import evaluate_emotion_classification, evaluate_generation
from emotion_steering.models.activations import determine_generation_phase
from emotion_steering.models.injection import (apply_activation_vector, gate_intervention_stage,
                                               scale_steering_vectors, validate_vector_hook_locations)
from emotion_steering.methods.vector import MeanDiffSteering
from emotion_steering.methods.vector import ProbeSteering
from emotion_steering.methods.peft import (AssistantOnlyCollator, assistant_only_features,
                                     prepare_dpo_records, validate_dpo_records,
                                     validate_sft_records)
from emotion_steering.methods.registry import METHODS, create_method
from emotion_steering.cli import resolve_run_all_config
from emotion_steering.experiments.generate import (dataset_fingerprint, experiment_signatures, prepare_steering,
                                                   run_records, validate_hook_execution)
from emotion_steering.methods.prompt import DescriptiveSteering, FewShotSteering, ZeroShotSteering
import emotion_steering.experiments.extract as extract_module
import emotion_steering.experiments.generate as generate_module


class FakeTokenizer:
    is_fast = True

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """验证apply chat template行为。"""
        user = messages[-1]["content"]
        return f"<system>irrelevant prompt</system><user>{user}</user><assistant>"

    def __call__(self, texts, padding=True, return_tensors="pt", return_offsets_mapping=True, truncation=False, add_special_tokens=True):
        """处理一次模型前向调用。"""
        if isinstance(texts, str): texts = [texts]
        ids, offsets = [], []
        for text in texts:
            ids.append([ord(char) for char in text])
            offsets.append([(i, i + 1) for i in range(len(text))])
        width = max(map(len, ids))
        mask = [[1] * len(row) + [0] * (width-len(row)) for row in ids]
        padded_ids = [row + [0] * (width-len(row)) for row in ids]
        padded_offsets = [row + [(0, 0)] * (width-len(row)) for row in offsets]
        return {"input_ids": torch.tensor(padded_ids), "attention_mask": torch.tensor(mask),
                "offset_mapping": torch.tensor(padded_offsets)}


class FakeBlock(nn.Module):
    def __init__(self):
        """初始化当前对象及其配置。"""
        super().__init__()
        self.hook_after_mlp_hs = nn.Identity()


class FakeModel(nn.Module):
    def __init__(self):
        """初始化当前对象及其配置。"""
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeBlock(), FakeBlock()])

    @property
    def device(self):
        """验证device行为。"""
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask, use_cache):
        """执行当前模型组件的前向计算。"""
        base = input_ids.float().unsqueeze(-1)
        hidden = torch.cat([base, base * 2, base * 3], dim=-1)
        for block in self.model.layers:
            hidden = block.hook_after_mlp_hs(hidden + 1)
        return hidden


def test_user_token_extraction_excludes_chat_template_tokens():
    """验证user token extraction excludes chat template tokens行为。"""
    rows = [{"text": "UserOnly"}, {"text": "Text"}]
    activations = extract_user_token_activations(rows, FakeTokenizer(), FakeModel(), batch_size=2)
    assert sorted(activations) == [0, 1]
    expected = torch.tensor([(sum(map(ord, "UserOnly")) + 8) / 8,
                             (2 * sum(map(ord, "UserOnly")) + 8) / 8,
                             (3 * sum(map(ord, "UserOnly")) + 8) / 8])
    assert torch.allclose(activations[0][0], expected)
    assert len(activations) == 2


def test_meandiff_and_probe_artifacts_have_expected_dimensions():
    """验证meandiff and probe artifacts have expected dimensions行为。"""
    positive = torch.tensor([[2., 2.], [4., 4.]])
    negative = torch.tensor([[0., 0.], [0., 2.]])
    artifact = MeanDiffSteering().fit(None, {"positive": positive, "negative": negative}, {})
    assert torch.allclose(artifact.payload, torch.tensor([3., 2.]))
    probe = ProbeSteering().fit(None, {"positive": positive, "negative": negative}, {"seed": 1})
    assert probe.payload["vector"].shape == (2,)
    assert 0 <= probe.metadata["train_accuracy"] <= 1


def test_vector_metadata_roundtrip_and_model_guard(tmp_path):
    """验证vector metadata roundtrip and model guard行为。"""
    save_vector_artifact(tmp_path / "vector", {2: torch.tensor([3., 4.])}, {"model": "toy"})
    vectors, metadata = load_vector_artifact(tmp_path / "vector", {"model": "toy"})
    assert metadata["vector_norms"]["2"] == 5
    assert torch.equal(vectors[2], torch.tensor([3., 4.]))
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_vector_artifact(tmp_path / "vector", {"model": "other"})


def test_method_registry_and_training_data_validation():
    """验证method registry and training data validation行为。"""
    assert {"zero_shot", "few_shot", "descriptive", "meandiff", "probe", "sft", "dpo"}.issubset(METHODS)
    assert create_method("zero_shot").artifact_type == "system_prompt"
    assert validate_sft_records([{"user": "help", "assistant": "reply"}])
    assert validate_dpo_records([{"user": "help", "chosen": "good", "rejected": "bad"}])
    with pytest.raises(ValueError, match="chosen and rejected"):
        validate_dpo_records([{"user": "help", "chosen": "same", "rejected": "same"}])
    with pytest.raises(ValueError, match="at least one"):
        validate_sft_records([])


def test_sft_masks_prompt_loss_and_pads_batches():
    """验证sft masks prompt loss and pads batches行为。"""
    features = assistant_only_features([1, 2, 3], [1, 2, 3, 4, 5])
    assert features["labels"] == [-100, -100, -100, 4, 5]
    collator = AssistantOnlyCollator(pad_token_id=0)
    batch = collator([features, assistant_only_features([8], [8, 9])])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5], [-100, 9, -100, -100, -100]]


def test_dpo_records_are_rendered_into_prompt_completion_fields():
    """验证dpo records are rendered into prompt completion fields行为。"""
    record = {"system": "be supportive", "user": "help", "chosen": "kind reply", "rejected": "rude reply"}
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_kwargs):
            """验证apply chat template行为。"""
            assert messages[-1]["role"] == "user"
            return "PROMPT:" + messages[-1]["content"] + ":ASSISTANT:"
    converted = prepare_dpo_records([record], Tokenizer())
    assert converted == [{"prompt": "PROMPT:help:ASSISTANT:", "chosen": "kind reply", "rejected": "rude reply"}]


def test_evaluation_separates_classification_from_generation():
    """验证evaluation separates classification from generation行为。"""
    result = evaluate_emotion_classification([{"label": 1, "prediction": 1}, {"label": 0, "prediction": 1}])
    assert result["task"] == "brighter_emotion_classification"
    assert "macro_f1" in result
    generation = evaluate_generation([{"response": "hello"}])
    assert generation["scoring_status"] == "not_scored"


def test_activation_intervention_modes_and_stage_gating():
    """验证activation intervention modes and stage gating行为。"""
    base = torch.ones((1, 3, 2))
    added = apply_activation_vector(base, torch.tensor([1., 0.]), coefficient=2, mode="add", token_indices=[-1])
    assert torch.equal(added[0, -1], torch.tensor([3., 1.]))
    replaced = apply_activation_vector(base, torch.tensor([2., 0.]), mode="replace", token_indices=[0])
    assert torch.equal(replaced[0, 0], torch.tensor([2., 0.]))
    assert torch.equal(replaced[0, 1], base[0, 1])
    hook = lambda x, _hook: x + 1
    prefill = gate_intervention_stage(hook, "prefill")
    assert torch.equal(prefill(base, None), base + 1)
    one_token = torch.ones((1, 1, 2))
    assert torch.equal(prefill(one_token, None), one_token)
    explicit_phase = gate_intervention_stage(hook, "prefill", phase_getter=lambda: "prefill")
    assert torch.equal(explicit_phase(one_token, None), one_token + 1)


def test_generation_phase_uses_cache_and_prompt_progress():
    """验证generation phase uses cache and prompt progress行为。"""
    assert determine_generation_phase(None, input_length=1, prompt_length=1) == "prefill"
    assert determine_generation_phase(None, input_length=2, prompt_length=1) == "generation"
    assert determine_generation_phase(object(), input_length=1, prompt_length=1) == "generation"
    class EmptyCache:
        """返回空缓存长度。"""
        def get_seq_length(self):
            """返回缓存中的词元数量。"""
            return 0
    assert determine_generation_phase(EmptyCache(), input_length=3, prompt_length=3) == "prefill"


def test_explicit_stage_gate_counts_only_requested_hooks():
    """验证explicit stage gate counts only requested hooks行为。"""
    phases = {"value": "prefill"}
    hook = lambda activations, _hook: activations + 1
    gated = gate_intervention_stage(hook, "both", phase_getter=lambda: phases["value"],
                                    target_names=["model.layers.2.hook_after_mlp_hs"])
    target = type("Hook", (), {"name": "model.layers.2.hook_after_mlp_hs"})()
    other = type("Hook", (), {"name": "model.layers.1.hook_after_mlp_hs"})()
    acts = torch.zeros((1, 1, 2))
    assert torch.equal(gated(acts, target), acts + 1)
    assert torch.equal(gated(acts, other), acts)
    phases["value"] = "generation"
    gated(acts, target)
    assert gated.call_counts == {"prefill": {target.name: 1}, "generation": {target.name: 1}}


def test_hook_validation_accepts_one_token_output_without_decode_forward():
    """验证hook validation accepts one token output without decode forward行为。"""
    calls = {"prefill": {"model.layers.2.hook_after_mlp_hs": 1}, "generation": {}}
    forwards = {"prefill": 1, "generation": 0}
    skipped = validate_hook_execution({"prefill", "generation"}, [2], calls, forwards)
    assert skipped == {"generation": "generation produced no forward pass in this phase"}


def test_hook_validation_fails_if_observed_phase_misses_target_hook():
    """验证hook validation fails if observed phase misses target hook行为。"""
    with pytest.raises(RuntimeError, match="observed prefill phase"):
        validate_hook_execution({"prefill", "generation"}, [2],
                                {"prefill": {}, "generation": {}},
                                {"prefill": 1, "generation": 0})


def test_generation_records_user_and_effective_system_prompt(tmp_path):
    """验证generation records user and effective system prompt行为。"""
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            """验证apply chat template行为。"""
            self.messages = messages
            return torch.tensor([[1, 2]])
        def decode(self, _tokens, **_kwargs):
            """验证decode行为。"""
            return "answer"
    class Model:
        device = torch.device("cpu")
        def generate(self, inputs, **_kwargs):
            """验证generate行为。"""
            return torch.cat([inputs, torch.tensor([[3]])], dim=1)

    tokenizer = Tokenizer()
    results = run_records(Model(), tokenizer, [{"sample_id": "s1", "user": "help me", "system": "be kind"}],
                          {"model": "toy", "method": "none", "generation": {}}, tmp_path / "out.jsonl")
    assert results[0]["user"] == "help me"
    assert results[0]["system"] == "be kind"
    assert results[0]["effective_system_prompt"].endswith("be kind")
    stored = json.loads((tmp_path / "out.jsonl").read_text(encoding="utf-8").strip())
    assert stored["user"] == "help me" and stored["effective_system_prompt"].endswith("be kind")


def test_normalization_preserves_requested_steering_strength():
    """验证normalization preserves requested steering strength行为。"""
    vectors = {0: torch.tensor([3., 4.])}
    unit_strength = scale_steering_vectors(vectors, [0], coefficient=1, normalize=True)
    five_strength = scale_steering_vectors(vectors, [0], coefficient=5, normalize=True)
    assert torch.allclose(unit_strength[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(five_strength[0], torch.tensor([3., 4.]))
    assert torch.isclose(five_strength.norm(), unit_strength.norm() * 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_activation_injection_accepts_cpu_vector():
    """验证gpu activation injection accepts cpu vector行为。"""
    activations = torch.zeros((1, 1, 2), device="cuda", dtype=torch.float16)
    result = apply_activation_vector(activations, torch.tensor([1., 2.]), coefficient=3)
    assert result.device.type == "cuda"
    assert torch.equal(result.cpu(), torch.tensor([[[3., 6.]]], dtype=torch.float16))


def test_hook_location_must_match_extraction_point():
    """验证hook location must match extraction point行为。"""
    metadata = {"hook_locations": [7]}
    assert validate_vector_hook_locations(metadata, [7]) == [7]
    with pytest.raises(ValueError, match="do not match"):
        validate_vector_hook_locations(metadata, [1])
    with pytest.raises(ValueError, match="only PsySET hook location 7"):
        validate_vector_hook_locations({"hook_locations": [1]}, [1])


def test_run_all_config_paths_resolve_from_project_root(tmp_path):
    """验证run all config paths resolve from project root行为。"""
    path = resolve_run_all_config("configs/extraction.yaml", project_root=tmp_path)
    assert path == tmp_path / "configs/extraction.yaml"
    absolute = tmp_path / "custom.yaml"
    assert resolve_run_all_config(str(absolute), project_root=tmp_path) == absolute


def test_experiment_signature_changes_with_method():
    config = {"model": "toy", "method": "meandiff", "target_emotion": "joy",
              "target_language": "chn", "source_language": "eng", "layers": [16],
              "coefficient": 1, "generation": {"do_sample": False}}
    first, generation = experiment_signatures(config)
    changed, same_generation = experiment_signatures({**config, "method": "probe"})
    assert first != changed
    assert generation == same_generation


def test_experiment_signature_includes_intervention_and_hook_settings(tmp_path):
    config = {"model": "toy", "method": "meandiff", "target_emotion": "joy",
              "target_language": "chn", "intervention": "add", "hook_locations": [7],
              "generation": {"do_sample": False}}
    original, _ = experiment_signatures(config)
    changed_intervention, _ = experiment_signatures({**config, "intervention": "replace"})
    changed_hook, _ = experiment_signatures({**config, "hook_locations": [2]})
    template = tmp_path / "prompt.yaml"
    template.write_text("one", encoding="utf-8")
    first_file, _ = experiment_signatures({**config, "prompt_templates_path": str(template)})
    template.write_text("two", encoding="utf-8")
    second_file, _ = experiment_signatures({**config, "prompt_templates_path": str(template)})
    assert len({original, changed_intervention, changed_hook, first_file, second_file}) == 5


def test_dataset_fingerprint_tracks_input_content(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text("first", encoding="utf-8")
    first = dataset_fingerprint(data)
    data.write_text("second", encoding="utf-8")
    assert first != dataset_fingerprint(data)


def test_run_records_rejects_duplicate_sample_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate sample_id"):
        run_records(object(), object(), [{"sample_id": "same"}, {"sample_id": "same"}],
                    {"model": "toy", "method": "none"}, tmp_path / "out.jsonl")


def test_prompt_templates_render_all_language_emotion_pairs():
    """验证十种语言的两类情绪提示都能完整渲染。"""
    languages = ["eng", "chn", "esp", "deu", "hin", "mar", "rus", "tat", "ron", "hau"]
    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    for language in languages:
        for emotion in emotions:
            config = {"target_language": language, "target_emotion": emotion}
            for method in (ZeroShotSteering(), DescriptiveSteering()):
                prompt = method.fit(None, None, config).payload
                assert prompt and "{" not in prompt and "}" not in prompt


def test_few_shot_requires_reviewed_unique_examples_and_records_ids(tmp_path):
    """验证 Few-shot 只选择已审核且编号唯一的示例。"""
    path = tmp_path / "examples.jsonl"
    examples = [{"example_id": f"joy_{i}", "scenario_id": f"scene_{i}", "language": "eng",
                 "target_emotion": "joy", "split": "few_shot", "translation_reviewed": True,
                 "order": i, "user": f"User {i}", "assistant": f"Reply {i}"} for i in range(3)]
    path.write_text("\n".join(json.dumps(row) for row in examples), encoding="utf-8")
    artifact = FewShotSteering().fit(None, None, {"target_language": "eng", "target_emotion": "joy",
                                                  "few_shot_examples_path": str(path),
                                                  "prompt": {"num_examples": 3}})
    assert artifact.metadata["example_ids"] == ["joy_0", "joy_1", "joy_2"]
    examples[0]["translation_reviewed"] = False
    path.write_text("\n".join(json.dumps(row) for row in examples), encoding="utf-8")
    with pytest.raises(ValueError, match="translation review"):
        FewShotSteering().fit(None, None, {"target_language": "eng", "target_emotion": "joy",
                                           "few_shot_examples_path": str(path)})


def test_probe_artifact_uses_shared_vector_injection_branch(tmp_path, monkeypatch):
    """验证probe artifact uses shared vector injection branch行为。"""
    vector_dir = tmp_path / "probe"
    save_vector_artifact(vector_dir, {3: torch.tensor([3., 4.])}, {
        "model": "toy-model", "source_language": "eng", "emotion": "joy",
        "method": "probe", "hook_locations": [7], "hidden_size": 2,
    })
    received = {}
    def fake_get_steer_fn(vectors, layers, locations, token_slice, hidden_size, mode, renormalize):
        """验证fake get steer fn行为。"""
        received.update(vectors=vectors, layers=layers, locations=locations, mode=mode)
        return lambda activations, hook: activations
    monkeypatch.setattr(generate_module, "get_steer_fn", fake_get_steer_fn)

    class Model:
        config = type("Config", (), {"hidden_size": 2})()
        """设置模型使用的激活干预函数。"""
        def set_steer_fn(self, fn):
            """保存模型的激活干预函数。"""
            self.steer_fn = fn
    model = Model()
    _, artifact = prepare_steering(model, {}, {
        "method": "probe", "vector_path": str(vector_dir), "model": "toy-model",
        "source_language": "eng", "emotion": "joy", "layers": [3],
        "hook_locations": [7], "normalize": True, "coefficient": 2.0,
    })
    assert artifact["method"] == "probe" and artifact["layers"] == [3]
    assert received["locations"] == [7]
    assert torch.allclose(received["vectors"].flatten(), torch.tensor([1.2, 1.6]))


def test_methods_share_one_activation_extraction_per_task(tmp_path, monkeypatch):
    """验证methods share one activation extraction per task行为。"""
    calls = {"extract": 0}
    loaded = {}
    monkeypatch.setattr(extract_module, "load_tokenizer", lambda _name: object())
    monkeypatch.setattr(extract_module, "load_model", lambda *_args, **_kwargs: object())
    def fake_load_brighter(language, **kwargs):
        """验证fake load brighter行为。"""
        loaded.update(language=language, **kwargs)
        return [], [], ["joy"], "multitask"
    monkeypatch.setattr(extract_module, "load_brighter_local", fake_load_brighter)
    frame = __import__("pandas").DataFrame({"id": [f"id{i}" for i in range(8)],
                                            "label": [0, 1] * 4})
    states = torch.tensor([[float(i), float(i % 3)] for i in range(8)])
    def prepare(*_args, **_kwargs):
        """验证prepare行为。"""
        calls["extract"] += 1
        return frame, {0: states}
    monkeypatch.setattr(extract_module, "extract_task_activations", prepare)
    config = {"model": "test/model", "run_id": "unit", "languages": ["eng"],
              "emotions": ["joy"], "methods": ["meandiff", "probe"],
              "output_dir": str(tmp_path / "vectors"), "seed": 7,
              "validation_fraction": 0.25, "data_root": str(tmp_path / "custom-brighter")}
    assert extract_module.run(config) == 2
    assert calls["extract"] == 1
    assert loaded["data_root"] == str(tmp_path / "custom-brighter")
    assert (tmp_path / "vectors/test_model/unit/meandiff/eng/joy/vectors.pt").exists()
    assert (tmp_path / "vectors/test_model/unit/probe/eng/joy/probe.pt").exists()
