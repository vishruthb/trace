"""Tests for audit module."""

from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

import trace_lib.optimization_auditor as optimization_auditor_module
import trace_lib.hook_manager as hook_manager_module
from trace_lib.audit import Audit
from trace_lib.sae import SparseAutoencoder


class _DummyConsole:
    def print(self, *args, **kwargs):
        pass


class _DummyBatch(dict):
    def to(self, device):
        return self


class TestAudit:
    def test_load_sae_directory(self, monkeypatch, tmp_path: Path):
        """load_sae should load all layer_* subdirectories via trace_lib.sae."""
        layer0 = tmp_path / "layer_0"
        layer3 = tmp_path / "layer_3"
        layer0.mkdir()
        layer3.mkdir()

        def fake_load(cls, path, device=None):
            return {"path": str(path), "device": device}

        monkeypatch.setattr(SparseAutoencoder, "load", classmethod(fake_load))

        audit = object.__new__(Audit)
        audit.device = "cpu"
        audit.sae_dict = {}
        audit.console = _DummyConsole()

        audit.load_sae(tmp_path)

        assert set(audit.sae_dict.keys()) == {0, 3}
        assert audit.sae_dict[0]["path"] == str(layer0)
        assert audit.sae_dict[3]["path"] == str(layer3)
        assert audit.sae_dict[0]["device"] == "cpu"

    def test_trace_features_uses_baseline_prompts_for_masking(self):
        """Baseline prompts should produce a safety mask that changes CPFR."""
        safety_prompts = ["harmful prompt"]
        baseline_prompts = ["benign prompt"]

        base_model = object()
        quant_model = object()

        features = {
            ("base", tuple(safety_prompts)): torch.tensor(
                [[10.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
            ),
            ("quant", tuple(safety_prompts)): torch.tensor(
                [[0.0, 1.0, 1.0, 1.0], [10.0, 1.0, 1.0, 1.0]]
            ),
            ("base", tuple(baseline_prompts)): torch.tensor(
                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
        }
        calls = []

        def fake_extract(self, model, prompts, layer, batch_size=4):
            key = ("base" if model is base_model else "quant", tuple(prompts))
            calls.append(key)
            return features[key]

        audit = object.__new__(Audit)
        audit.base_model = base_model
        audit.quant_model = quant_model
        audit.base_model_name = "base"
        audit.quant_model_name = "quant"
        audit.sae_dict = {0: SimpleNamespace(config=SimpleNamespace(d_sae=4))}
        audit.activation_device = "cpu"
        audit._extract_features = MethodType(fake_extract, audit)

        masked = audit.trace_features(
            dataset=safety_prompts,
            baseline_prompts=baseline_prompts,
            layers=[0],
            safety_z_threshold=1.0,
            min_safety_features=1,
        )
        unmasked = audit.trace_features(
            dataset=safety_prompts,
            baseline_prompts=[],
            layers=[0],
            safety_z_threshold=1.0,
            min_safety_features=1,
        )

        assert ("base", tuple(baseline_prompts)) in calls
        assert masked.cpfr_score < unmasked.cpfr_score

    def test_run_precision_sweep_forwards_baseline_and_activation_device(self, monkeypatch):
        """run_precision_sweep should pass baseline and activation-device settings."""
        captured = {}

        class FakeOptimizationAuditor:
            def __init__(self, model, sae_dict, tokenizer, device, activation_device):
                captured["init"] = {
                    "model": model,
                    "sae_dict": sae_dict,
                    "tokenizer": tokenizer,
                    "device": device,
                    "activation_device": activation_device,
                }

            def run_sweep(
                self,
                test_prompts,
                baseline_prompts=None,
                configs=None,
                model_name="",
                safety_z_threshold=1.0,
                min_safety_features=1,
            ):
                captured["run_sweep"] = {
                    "test_prompts": test_prompts,
                    "baseline_prompts": baseline_prompts,
                    "configs": configs,
                    "model_name": model_name,
                    "safety_z_threshold": safety_z_threshold,
                    "min_safety_features": min_safety_features,
                }
                return "ok"

        monkeypatch.setattr(
            optimization_auditor_module,
            "OptimizationAuditor",
            FakeOptimizationAuditor,
        )

        audit = object.__new__(Audit)
        audit.base_model = object()
        audit.sae_dict = {0: object()}
        audit.tokenizer = object()
        audit.device = "cpu"
        audit.activation_device = "cuda"
        audit.base_model_name = "demo-model"

        result = audit.run_precision_sweep(
            test_prompts=["safety"],
            baseline_prompts=["baseline"],
            configs=["cfg"],
            safety_z_threshold=1.5,
            min_safety_features=2,
        )

        assert result == "ok"
        assert captured["init"]["activation_device"] == "cuda"
        assert captured["run_sweep"]["baseline_prompts"] == ["baseline"]
        assert captured["run_sweep"]["safety_z_threshold"] == 1.5
        assert captured["run_sweep"]["min_safety_features"] == 2

    def test_extract_features_uses_configured_activation_device(self, monkeypatch):
        """_extract_features should use audit.activation_device for hook storage."""
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
            def __call__(self, **kwargs):
                return None

        class FakeSAE:
            def __init__(self):
                self.config = SimpleNamespace(device="cpu", d_sae=3)

            def encode(self, x):
                return x

        def fake_tokenizer(*args, **kwargs):
            return _DummyBatch({"input_ids": torch.tensor([[1, 2, 3]])})

        audit = object.__new__(Audit)
        audit.device = "cpu"
        audit.activation_device = "cuda"
        audit.tokenizer = fake_tokenizer
        audit.sae_dict = {0: FakeSAE()}

        features = audit._extract_features(FakeModel(), ["hello"], layer=0)

        assert features.shape == (1, 3)
        assert captured_devices == ["cuda"]
