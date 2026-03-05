"""Tests for optimization_auditor module."""

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import torch

import trace_lib.hook_manager as hook_manager_module
from trace_lib.metrics import CPFR, FeatureDrift
from trace_lib.optimization_auditor import OptimizationAuditor, QuantizationConfig, QuantizationType


class TestOptimizationAuditor:
    def test_audit_config_uses_baseline_prompts_for_feature_masking(self):
        """baseline_prompts should impact CPFR via safety feature masking."""
        safety_prompts = ["harmful prompt"]
        baseline_prompts = ["benign prompt"]

        base_model = object()
        quant_model = object()

        feature_map = {
            ("base", tuple(safety_prompts)): {0: torch.tensor([[10.0, 1.0, 1.0, 1.0]])},
            ("quant", tuple(safety_prompts)): {0: torch.tensor([[2.0, 1.0, 1.0, 1.0]])},
            ("base", tuple(baseline_prompts)): {0: torch.tensor([[1.0, 1.0, 1.0, 1.0]])},
        }
        calls = []

        def fake_quantize_model(self, config):
            return quant_model

        def fake_get_model_memory_mb(self, model):
            return 100.0 if model is base_model else 50.0

        def fake_get_features_for_model(self, model, prompts, batch_size=4):
            key = ("base" if model is base_model else "quant", tuple(prompts))
            calls.append(key)
            return feature_map[key]

        auditor = object.__new__(OptimizationAuditor)
        auditor.model = base_model
        auditor.sae_dict = {0: SimpleNamespace(config=SimpleNamespace(device="cpu"))}
        auditor.cpfr_metric = CPFR(normalize=False)
        auditor.drift_metric = FeatureDrift()
        auditor.device = "cpu"
        auditor.activation_device = "cpu"
        auditor.quantize_model = MethodType(fake_quantize_model, auditor)
        auditor._get_model_memory_mb = MethodType(fake_get_model_memory_mb, auditor)
        auditor._get_features_for_model = MethodType(fake_get_features_for_model, auditor)

        config = QuantizationConfig(quant_type=QuantizationType.NONE)

        unmasked = auditor.audit_config(
            config=config,
            test_prompts=safety_prompts,
            baseline_prompts=None,
        )
        masked = auditor.audit_config(
            config=config,
            test_prompts=safety_prompts,
            baseline_prompts=baseline_prompts,
            safety_z_threshold=1.0,
            min_safety_features=1,
        )

        assert ("base", tuple(baseline_prompts)) in calls
        assert masked.cpfr_score < unmasked.cpfr_score

    def test_get_features_for_model_uses_configured_activation_device(self, monkeypatch):
        """_get_features_for_model should use activation_device for hook storage."""
        captured_devices = []

        class FakeHookManager:
            def __init__(self, model, config):
                captured_devices.append(config.device)

            @contextmanager
            def capture(self):
                yield

            def get_activations(self, hook_point, layer):
                return torch.ones(1, 2, 3)

        monkeypatch.setattr(hook_manager_module, "HookManager", FakeHookManager)

        class FakeModel:
            def __call__(self, input_ids):
                return None

        def fake_tokenizer(*args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        class FakeSAE:
            def __init__(self):
                self.config = SimpleNamespace(device="cpu")

            def encode(self, x):
                return x

        auditor = object.__new__(OptimizationAuditor)
        auditor.sae_dict = {0: FakeSAE()}
        auditor.tokenizer = fake_tokenizer
        auditor.device = "cpu"
        auditor.activation_device = "cuda"

        out = auditor._get_features_for_model(FakeModel(), ["prompt"])

        assert 0 in out
        assert out[0].shape == (1, 3)
        assert captured_devices == ["cuda"]
