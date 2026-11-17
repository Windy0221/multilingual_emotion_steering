"""测试文件：检查项目功能是否符合预期。"""

import torch

from emotion_steering.models.injection import EXTRACTION_LOCATIONS, get_steer_fn, name_to_loc_and_layer


def test_hook_location_mapping_matches_block_output_hook():
    """验证hook location mapping matches block output hook行为。"""
    assert EXTRACTION_LOCATIONS[7] == "model.layers.[LID].hook_after_mlp_hs"
    assert name_to_loc_and_layer("model.layers.3.hook_after_mlp_hs") == (7, 3)


def test_injection_adds_vector_to_selected_layer_and_token():
    """验证injection adds vector to selected layer and token行为。"""
    vector = torch.tensor([[[[2.0, -1.0]]]])
    steer = get_steer_fn(vector, [3], [7], torch.tensor([-1]), 2, "add", False)
    hook = type("Hook", (), {"name": "model.layers.3.hook_after_mlp_hs"})()
    activations = torch.zeros((1, 2, 2))
    result = steer(activations, hook)
    assert torch.equal(result, torch.tensor([[[0.0, 0.0], [2.0, -1.0]]]))
    assert torch.equal(activations, torch.zeros((1, 2, 2)))
