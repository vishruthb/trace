"""
CircuitTracer: Maps safety behaviors to their corresponding SAE features.

Creates a "Safety Fingerprint" of the model by identifying which sparse
features activate during specific safety-critical behaviors (e.g., refusal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from trace_lib.hook_manager import HookManager, HookConfig
from trace_lib.sae import SparseAutoencoder


@dataclass
class SafetyFeature:
    """Represents a single safety-critical feature."""

    feature_id: int
    layer: int
    description: str = ""
    activation_threshold: float = 0.1

    # Feature characteristics
    mean_activation: float = 0.0
    max_activation: float = 0.0
    activation_frequency: float = 0.0  # How often it fires

    # Behavioral associations
    associated_behaviors: list[str] = field(default_factory=list)
    example_prompts: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SafetyFeature(id={self.feature_id}, layer={self.layer}, "
            f"desc='{self.description[:30]}...')"
        )


@dataclass
class SafetyFingerprint:
    """A collection of safety features that characterize a model's safety circuits."""

    features: list[SafetyFeature] = field(default_factory=list)
    model_name: str = ""
    creation_timestamp: str = ""

    # Feature statistics
    num_features: int = 0
    layers_covered: list[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"SafetyFingerprint(features={len(self.features)}, model='{self.model_name}')"

    def get_features_by_layer(self, layer: int) -> list[SafetyFeature]:
        """Get all safety features at a specific layer."""
        return [f for f in self.features if f.layer == layer]

    def get_features_by_behavior(self, behavior: str) -> list[SafetyFeature]:
        """Get features associated with a specific behavior."""
        return [f for f in self.features if behavior in f.associated_behaviors]

    def to_dict(self) -> dict:
        """Convert fingerprint to dictionary."""
        return {
            "model_name": self.model_name,
            "creation_timestamp": self.creation_timestamp,
            "num_features": self.num_features,
            "layers_covered": self.layers_covered,
            "features": [
                {
                    "feature_id": f.feature_id,
                    "layer": f.layer,
                    "description": f.description,
                    "activation_threshold": f.activation_threshold,
                    "mean_activation": f.mean_activation,
                    "max_activation": f.max_activation,
                    "activation_frequency": f.activation_frequency,
                    "associated_behaviors": f.associated_behaviors,
                    "example_prompts": f.example_prompts,
                }
                for f in self.features
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyFingerprint":
        """Create fingerprint from dictionary."""
        fingerprint = cls(
            model_name=data["model_name"],
            creation_timestamp=data.get("creation_timestamp", ""),
            num_features=data["num_features"],
            layers_covered=data["layers_covered"],
        )

        for f_data in data["features"]:
            feature = SafetyFeature(
                feature_id=f_data["feature_id"],
                layer=f_data["layer"],
                description=f_data.get("description", ""),
                activation_threshold=f_data.get("activation_threshold", 0.1),
                mean_activation=f_data.get("mean_activation", 0.0),
                max_activation=f_data.get("max_activation", 0.0),
                activation_frequency=f_data.get("activation_frequency", 0.0),
                associated_behaviors=f_data.get("associated_behaviors", []),
                example_prompts=f_data.get("example_prompts", []),
            )
            fingerprint.features.append(feature)

        return fingerprint

    def save(self, path: Union[str, Path]):
        """Save fingerprint to file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SafetyFingerprint":
        """Load fingerprint from file."""
        import json

        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class CircuitTracerConfig:
    """Configuration for CircuitTracer."""

    # Layers to analyze
    layers: Union[list[int], str] = "all"

    # Minimum activation to consider a feature "active"
    activation_threshold: float = 0.1

    # Minimum activation frequency to include in fingerprint
    min_frequency: float = 0.01

    # Top-k features to track per behavior
    top_k_features: int = 100

    # Device and dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class CircuitTracer:
    """
    Maps safety behaviors to their corresponding SAE features.

    Creates a "Safety Fingerprint" that captures which features are
    characteristic of safety-critical model behaviors.

    Example:
        tracer = CircuitTracer(model, sae_dict)
        fingerprint = tracer.create_fingerprint(
            safety_prompts=refusal_prompts,
            baseline_prompts=normal_prompts,
        )
    """

    # Predefined safety behavior categories
    SAFETY_BEHAVIORS = [
        "refusal",           # Refusing harmful requests
        "uncertainty",       # Expressing uncertainty
        "clarification",     # Asking for clarification
        "safety_warning",    # Warning about dangerous content
        "ethical_reasoning", # Discussing ethics
        "source_citation",   # Citing sources/being factual
    ]

    def __init__(
        self,
        model: nn.Module,
        sae_dict: dict[int, SparseAutoencoder],
        config: Optional[CircuitTracerConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize CircuitTracer.

        Args:
            model: The model to trace
            sae_dict: Dictionary mapping layer -> SAE for that layer
            config: Tracer configuration
            tokenizer: Tokenizer for the model (optional)
        """
        self.model = model
        self.sae_dict = sae_dict
        self.config = config or CircuitTracerConfig()
        self.tokenizer = tokenizer

        # Set up hook manager
        hook_config = HookConfig(
            layers=list(sae_dict.keys()) if isinstance(self.config.layers, str) else self.config.layers,
            hook_points=["resid_post"],
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self.hook_manager = HookManager(model, hook_config)

    def _get_features_for_prompts(
        self,
        prompts: list[str],
        batch_size: int = 4,
    ) -> dict[int, torch.Tensor]:
        """
        Get SAE features for a list of prompts.

        Args:
            prompts: List of text prompts
            batch_size: Batch size for processing

        Returns:
            Dict mapping layer -> features tensor (n_prompts, d_sae)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for processing text prompts")

        all_features = {layer: [] for layer in self.sae_dict.keys()}

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(self.config.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.config.device)

            # Get activations
            with self.hook_manager.capture():
                with torch.no_grad():
                    self.model(input_ids, attention_mask=attention_mask)

            # Extract features through SAEs
            for layer, sae in self.sae_dict.items():
                activations = self.hook_manager.get_activations("resid_post", layer=layer)

                if isinstance(activations, list) and len(activations) == 0:
                    continue

                # Take mean over sequence dimension
                activations = activations.mean(dim=1)  # (batch, d_model)

                # Get SAE features
                features = sae.encode(activations.to(sae.config.device))
                all_features[layer].append(features.cpu())

        # Concatenate all batches
        result = {}
        for layer, feature_list in all_features.items():
            if feature_list:
                result[layer] = torch.cat(feature_list, dim=0)

        return result

    def identify_safety_features(
        self,
        safety_prompts: list[str],
        baseline_prompts: list[str],
        behavior_name: str = "safety",
        significance_threshold: float = 2.0,
    ) -> dict[int, list[int]]:
        """
        Identify features that are significantly more active for safety prompts.

        Uses a contrastive approach: features that activate much more strongly
        for safety prompts compared to baseline prompts are flagged.

        Args:
            safety_prompts: Prompts that should trigger safety behavior
            baseline_prompts: Normal prompts for comparison
            behavior_name: Name of the safety behavior being traced
            significance_threshold: How many std devs above baseline to be "significant"

        Returns:
            Dict mapping layer -> list of significant feature indices
        """
        # Get features for both prompt sets
        safety_features = self._get_features_for_prompts(safety_prompts)
        baseline_features = self._get_features_for_prompts(baseline_prompts)

        significant_features = {}

        for layer in self.sae_dict.keys():
            if layer not in safety_features or layer not in baseline_features:
                continue

            sf = safety_features[layer]  # (n_safety, d_sae)
            bf = baseline_features[layer]  # (n_baseline, d_sae)

            # Compute mean and std for baseline
            baseline_mean = bf.mean(dim=0)
            baseline_std = bf.std(dim=0) + 1e-8

            # Compute mean for safety
            safety_mean = sf.mean(dim=0)

            # Find features with significantly higher activation
            z_scores = (safety_mean - baseline_mean) / baseline_std
            significant_mask = z_scores > significance_threshold

            significant_features[layer] = significant_mask.nonzero(as_tuple=True)[0].tolist()

        return significant_features

    def create_fingerprint(
        self,
        safety_prompts: dict[str, list[str]],
        baseline_prompts: list[str],
        model_name: str = "",
    ) -> SafetyFingerprint:
        """
        Create a comprehensive safety fingerprint for the model.

        Args:
            safety_prompts: Dict mapping behavior name -> list of prompts
            baseline_prompts: Normal prompts for comparison
            model_name: Name to assign to the fingerprint

        Returns:
            SafetyFingerprint containing all identified safety features
        """
        import datetime

        fingerprint = SafetyFingerprint(
            model_name=model_name,
            creation_timestamp=datetime.datetime.now().isoformat(),
            layers_covered=list(self.sae_dict.keys()),
        )

        # Process each safety behavior
        for behavior, prompts in tqdm(safety_prompts.items(), desc="Creating fingerprint"):
            # Identify features for this behavior
            features_by_layer = self.identify_safety_features(
                safety_prompts=prompts,
                baseline_prompts=baseline_prompts,
                behavior_name=behavior,
            )

            # Get detailed feature statistics
            behavior_features = self._get_features_for_prompts(prompts)

            for layer, feature_ids in features_by_layer.items():
                if layer not in behavior_features:
                    continue

                features_tensor = behavior_features[layer]

                for fid in feature_ids[:self.config.top_k_features]:
                    feature_activations = features_tensor[:, fid]

                    safety_feature = SafetyFeature(
                        feature_id=fid,
                        layer=layer,
                        description=f"Feature {fid} at layer {layer} for {behavior}",
                        mean_activation=feature_activations.mean().item(),
                        max_activation=feature_activations.max().item(),
                        activation_frequency=(feature_activations > self.config.activation_threshold).float().mean().item(),
                        associated_behaviors=[behavior],
                        example_prompts=prompts[:3],  # Store a few example prompts
                    )

                    # Check if this feature is already in the fingerprint
                    existing = next(
                        (f for f in fingerprint.features if f.feature_id == fid and f.layer == layer),
                        None,
                    )
                    if existing:
                        existing.associated_behaviors.append(behavior)
                    else:
                        fingerprint.features.append(safety_feature)

        fingerprint.num_features = len(fingerprint.features)
        return fingerprint

    def trace_feature_activation(
        self,
        prompt: str,
        feature_ids: Optional[dict[int, list[int]]] = None,
    ) -> dict[int, dict[int, float]]:
        """
        Trace the activation of specific features for a single prompt.

        Args:
            prompt: The prompt to analyze
            feature_ids: Dict mapping layer -> feature IDs to trace
                        If None, traces all features

        Returns:
            Dict mapping layer -> (feature_id -> activation value)
        """
        features = self._get_features_for_prompts([prompt])

        result = {}
        for layer, feats in features.items():
            feats = feats.squeeze(0)  # Remove batch dim

            if feature_ids is not None and layer in feature_ids:
                # Only return specified features
                result[layer] = {
                    fid: feats[fid].item()
                    for fid in feature_ids[layer]
                    if fid < feats.shape[0]
                }
            else:
                # Return all non-zero features
                active_mask = feats > self.config.activation_threshold
                active_ids = active_mask.nonzero(as_tuple=True)[0]
                result[layer] = {
                    fid.item(): feats[fid].item()
                    for fid in active_ids
                }

        return result

    def compare_feature_activation(
        self,
        prompt: str,
        model_quant: nn.Module,
        fingerprint: SafetyFingerprint,
    ) -> dict:
        """
        Compare feature activations between reference and quantized models.

        Args:
            prompt: Prompt to test
            model_quant: Quantized model
            fingerprint: Safety fingerprint to check against

        Returns:
            Comparison results for each safety feature
        """
        # Get features from reference model (self.model)
        ref_features = self._get_features_for_prompts([prompt])

        # Set up tracer for quantized model
        quant_tracer = CircuitTracer(
            model_quant,
            self.sae_dict,
            self.config,
            self.tokenizer,
        )
        quant_features = quant_tracer._get_features_for_prompts([prompt])

        # Compare each safety feature
        comparisons = []
        for sf in fingerprint.features:
            layer = sf.layer
            fid = sf.feature_id

            if layer not in ref_features or layer not in quant_features:
                continue

            ref_act = ref_features[layer][0, fid].item()
            quant_act = quant_features[layer][0, fid].item()

            comparison = {
                "feature_id": fid,
                "layer": layer,
                "behaviors": sf.associated_behaviors,
                "ref_activation": ref_act,
                "quant_activation": quant_act,
                "degradation": 1.0 - (quant_act / (ref_act + 1e-8)) if ref_act > 0 else 0.0,
            }
            comparisons.append(comparison)

        return {
            "comparisons": comparisons,
            "degraded_features": [c for c in comparisons if c["degradation"] > 0.2],
            "lost_features": [c for c in comparisons if c["degradation"] > 0.8],
        }
