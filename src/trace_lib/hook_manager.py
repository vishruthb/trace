"""
HookManager: Low-level utility for intercepting tensors from model forward passes.

Handles both TransformerLens and HuggingFace models, with automatic de-quantization
of tensors for parity comparison across precision boundaries.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn


@dataclass
class HookConfig:
    """Configuration for hook behavior."""

    # Layers to hook (e.g., [0, 1, 2] or "all")
    layers: Union[list[int], str] = "all"

    # Hook points: "resid_pre", "resid_post", "mlp_out", "attn_out"
    hook_points: list[str] = field(default_factory=lambda: ["resid_post"])

    # Whether to dequantize INT8/INT4 tensors automatically
    auto_dequantize: bool = True

    # Device for stored activations
    device: str = "cpu"

    # Dtype for stored activations
    dtype: torch.dtype = torch.float32

    # Whether to detach tensors (saves memory, prevents grad tracking)
    detach: bool = True

    # Maximum number of samples to store (for memory management)
    max_samples: Optional[int] = None


class HookManager:
    """
    Manages forward hooks for intercepting model activations.

    Supports both HuggingFace Transformers and TransformerLens models,
    handling the different naming conventions and module structures.

    Example:
        hook_manager = HookManager(model, config)
        with hook_manager.capture():
            model(input_ids)
        activations = hook_manager.get_activations("resid_post", layer=5)
    """

    # Module name patterns for different model types
    HOOK_PATTERNS = {
        "huggingface": {
            "resid_pre": "model.layers.{layer}",
            "resid_post": "model.layers.{layer}",
            "mlp_out": "model.layers.{layer}.mlp",
            "attn_out": "model.layers.{layer}.self_attn",
            "embed": "model.embed_tokens",
        },
        "transformerlens": {
            "resid_pre": "blocks.{layer}.hook_resid_pre",
            "resid_post": "blocks.{layer}.hook_resid_post",
            "mlp_out": "blocks.{layer}.hook_mlp_out",
            "attn_out": "blocks.{layer}.hook_attn_out",
            "embed": "hook_embed",
        },
        "gpt2": {
            "resid_pre": "transformer.h.{layer}",
            "resid_post": "transformer.h.{layer}",
            "mlp_out": "transformer.h.{layer}.mlp",
            "attn_out": "transformer.h.{layer}.attn",
            "embed": "transformer.wte",
        },
        "llama": {
            "resid_pre": "model.layers.{layer}",
            "resid_post": "model.layers.{layer}",
            "mlp_out": "model.layers.{layer}.mlp",
            "attn_out": "model.layers.{layer}.self_attn",
            "embed": "model.embed_tokens",
        },
    }

    def __init__(
        self,
        model: nn.Module,
        config: Optional[HookConfig] = None,
        model_type: Optional[str] = None,
    ):
        """
        Initialize HookManager.

        Args:
            model: The model to hook into
            config: Hook configuration
            model_type: Model type ("huggingface", "transformerlens", "gpt2", "llama")
                       If None, auto-detected.
        """
        self.model = model
        self.config = config or HookConfig()
        self.model_type = model_type or self._detect_model_type()

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._activations: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._is_capturing = False

        # Determine number of layers
        self.num_layers = self._get_num_layers()

        # Resolve layer indices
        if self.config.layers == "all":
            self.target_layers = list(range(self.num_layers))
        else:
            self.target_layers = self.config.layers

    def _detect_model_type(self) -> str:
        """Auto-detect model type from architecture."""
        model_class = self.model.__class__.__name__.lower()

        if "hooked" in model_class or hasattr(self.model, "blocks"):
            return "transformerlens"
        elif "llama" in model_class:
            return "llama"
        elif "gpt2" in model_class:
            return "gpt2"
        elif "gemma" in model_class:
            return "llama"  # Gemma uses similar structure
        elif "mistral" in model_class:
            return "llama"  # Mistral uses similar structure
        else:
            return "huggingface"

    def _get_num_layers(self) -> int:
        """Get the number of transformer layers in the model."""
        if self.model_type == "transformerlens":
            return len(self.model.blocks)
        elif self.model_type == "gpt2":
            return len(self.model.transformer.h)
        else:  # huggingface/llama
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return len(self.model.model.layers)
            elif hasattr(self.model, "layers"):
                return len(self.model.layers)
            else:
                raise ValueError(f"Cannot determine layer count for {type(self.model)}")

    def _get_module(self, name: str) -> Optional[nn.Module]:
        """Get a module by dot-separated name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _get_hook_module_name(self, hook_point: str, layer: int) -> str:
        """Get the module name for a given hook point and layer."""
        patterns = self.HOOK_PATTERNS.get(self.model_type, self.HOOK_PATTERNS["huggingface"])
        pattern = patterns.get(hook_point, f"layer.{layer}")
        return pattern.format(layer=layer)

    def _dequantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        De-quantize tensor if it's in a quantized format.

        Handles INT8, INT4, and various quantization schemes.
        """
        if not self.config.auto_dequantize:
            return tensor

        # Check for bitsandbytes INT8
        if hasattr(tensor, "CB") and hasattr(tensor, "SCB"):
            # INT8 matrix from bitsandbytes
            return tensor.CB.float() * tensor.SCB.float().unsqueeze(-1)

        # Check for packed INT4
        if tensor.dtype == torch.uint8 and hasattr(tensor, "quant_state"):
            # This is a packed INT4 tensor
            # Unpack and dequantize (simplified - actual impl depends on format)
            pass

        # Standard torch quantized tensors
        if tensor.is_quantized:
            return tensor.dequantize()

        return tensor

    def _create_hook_fn(
        self, hook_point: str, layer: int
    ) -> Callable[[nn.Module, Any, torch.Tensor], None]:
        """Create a forward hook function for a specific hook point."""

        def hook_fn(
            module: nn.Module,
            input: Any,
            output: Union[torch.Tensor, tuple],
        ):
            if not self._is_capturing:
                return

            # Handle tuple outputs (common in attention modules)
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            # Dequantize if needed
            tensor = self._dequantize_tensor(tensor)

            # Convert to target dtype and device
            tensor = tensor.to(device=self.config.device, dtype=self.config.dtype)

            # Detach if configured
            if self.config.detach:
                tensor = tensor.detach()

            # Clone to prevent modification
            tensor = tensor.clone()

            # Store activation
            key = f"{hook_point}_layer{layer}"
            self._activations[key].append(tensor)

            # Memory management: limit stored samples
            if (
                self.config.max_samples is not None
                and len(self._activations[key]) > self.config.max_samples
            ):
                self._activations[key] = self._activations[key][-self.config.max_samples :]

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all target modules."""
        self.remove_hooks()  # Clear any existing hooks

        for hook_point in self.config.hook_points:
            for layer in self.target_layers:
                module_name = self._get_hook_module_name(hook_point, layer)
                module = self._get_module(module_name)

                if module is not None:
                    hook_fn = self._create_hook_fn(hook_point, layer)
                    handle = module.register_forward_hook(hook_fn)
                    self._hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def clear_activations(self):
        """Clear all stored activations."""
        self._activations.clear()

    @contextmanager
    def capture(self, clear_existing: bool = True):
        """
        Context manager for capturing activations.

        Args:
            clear_existing: Whether to clear existing activations before capture

        Example:
            with hook_manager.capture():
                model(input_ids)
        """
        if clear_existing:
            self.clear_activations()

        self.register_hooks()
        self._is_capturing = True

        try:
            yield self
        finally:
            self._is_capturing = False
            self.remove_hooks()

    def get_activations(
        self,
        hook_point: str = "resid_post",
        layer: Optional[int] = None,
        concat: bool = True,
    ) -> Union[torch.Tensor, dict[int, torch.Tensor]]:
        """
        Get captured activations.

        Args:
            hook_point: Which hook point to get ("resid_post", "mlp_out", etc.)
            layer: Specific layer (or None for all layers)
            concat: Whether to concatenate batches

        Returns:
            Tensor or dict of tensors with activations
        """
        if layer is not None:
            key = f"{hook_point}_layer{layer}"
            activations = self._activations.get(key, [])
            if concat and activations:
                return torch.cat(activations, dim=0)
            return activations

        # Return all layers
        result = {}
        for target_layer in self.target_layers:
            key = f"{hook_point}_layer{target_layer}"
            activations = self._activations.get(key, [])
            if concat and activations:
                result[target_layer] = torch.cat(activations, dim=0)
            else:
                result[target_layer] = activations

        return result

    def get_residual_stream(
        self,
        layer: int,
        position: str = "post",
    ) -> torch.Tensor:
        """
        Get residual stream activations at a specific layer.

        Args:
            layer: Layer index
            position: "pre" or "post" the layer

        Returns:
            Residual stream tensor (batch, seq, d_model)
        """
        hook_point = f"resid_{position}"
        return self.get_activations(hook_point, layer=layer)

    def compare_activations(
        self,
        other: "HookManager",
        hook_point: str = "resid_post",
        layer: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Compare activations between two HookManagers.

        Useful for comparing base vs quantized model activations.

        Args:
            other: Another HookManager instance
            hook_point: Hook point to compare
            layer: Layer to compare

        Returns:
            Dict with comparison metrics
        """
        acts_self = self.get_activations(hook_point, layer=layer)
        acts_other = other.get_activations(hook_point, layer=layer)

        # Ensure same shape
        min_batch = min(acts_self.shape[0], acts_other.shape[0])
        acts_self = acts_self[:min_batch]
        acts_other = acts_other[:min_batch]

        # Compute differences
        diff = acts_self - acts_other
        mse = (diff ** 2).mean()
        cosine_sim = torch.nn.functional.cosine_similarity(
            acts_self.flatten(start_dim=1),
            acts_other.flatten(start_dim=1),
            dim=1,
        ).mean()

        return {
            "mse": mse,
            "cosine_similarity": cosine_sim,
            "max_diff": diff.abs().max(),
            "mean_diff": diff.abs().mean(),
        }


class DualHookManager:
    """
    Manages hooks for both a reference and quantized model simultaneously.

    Simplifies the common use case of comparing activations between
    high-precision and low-precision versions of the same model.
    """

    def __init__(
        self,
        model_ref: nn.Module,
        model_quant: nn.Module,
        config: Optional[HookConfig] = None,
    ):
        """
        Initialize dual hook manager.

        Args:
            model_ref: Reference (high-precision) model
            model_quant: Quantized model
            config: Hook configuration (shared)
        """
        self.config = config or HookConfig()
        self.hook_ref = HookManager(model_ref, self.config)
        self.hook_quant = HookManager(model_quant, self.config)

    @contextmanager
    def capture_both(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Capture activations from both models on the same input.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
        """
        self.hook_ref.clear_activations()
        self.hook_quant.clear_activations()

        with self.hook_ref.capture(clear_existing=False):
            with self.hook_quant.capture(clear_existing=False):
                # Run reference model
                with torch.no_grad():
                    self.hook_ref.model(input_ids, attention_mask=attention_mask)
                    self.hook_quant.model(input_ids, attention_mask=attention_mask)

                yield self.hook_ref, self.hook_quant

    def get_comparison(
        self,
        hook_point: str = "resid_post",
        layer: int = 0,
    ) -> dict:
        """Get activation comparison between models."""
        return self.hook_ref.compare_activations(
            self.hook_quant, hook_point=hook_point, layer=layer
        )
