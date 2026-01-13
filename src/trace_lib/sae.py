"""
Sparse Autoencoder (SAE) implementation for feature extraction.

The SAE learns to decompose model activations into interpretable,
sparse features that can be tracked across precision boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder."""

    d_model: int  # Dimension of input activations
    d_sae: int  # Dimension of SAE latent space (typically 4x-16x d_model)
    l1_coefficient: float = 1e-3  # Sparsity penalty coefficient
    tied_weights: bool = False  # Whether decoder = encoder.T
    normalize_decoder: bool = True  # Unit norm decoder columns
    dtype: torch.dtype = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for extracting interpretable features from model activations.

    Encoder-Decoder Formulation:
        f(x) = ReLU(W_enc @ (x - b_dec) + b_enc)
        x_hat = W_dec @ f(x) + b_dec

    where f(x) represents the activated features (sparse representation).
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # Encoder weights and bias
        self.W_enc = nn.Parameter(
            torch.empty(config.d_sae, config.d_model, dtype=config.dtype, device=config.device)
        )
        self.b_enc = nn.Parameter(
            torch.zeros(config.d_sae, dtype=config.dtype, device=config.device)
        )

        # Decoder weights and bias
        if config.tied_weights:
            self.W_dec = None  # Use W_enc.T
        else:
            self.W_dec = nn.Parameter(
                torch.empty(config.d_model, config.d_sae, dtype=config.dtype, device=config.device)
            )
        self.b_dec = nn.Parameter(
            torch.zeros(config.d_model, dtype=config.dtype, device=config.device)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        if self.W_dec is not None:
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

        if self.config.normalize_decoder:
            self._normalize_decoder()

    def _normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        with torch.no_grad():
            W_dec = self.decoder_weights
            norms = torch.norm(W_dec, dim=0, keepdim=True)
            if self.W_dec is not None:
                self.W_dec.data = W_dec / (norms + 1e-8)
            else:
                # For tied weights, normalize encoder rows
                self.W_enc.data = self.W_enc / (
                    torch.norm(self.W_enc, dim=1, keepdim=True) + 1e-8
                )

    @property
    def decoder_weights(self) -> torch.Tensor:
        """Get decoder weight matrix (handles tied weights case)."""
        if self.config.tied_weights:
            return self.W_enc.T
        return self.W_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Sparse feature activations of shape (..., d_sae)
        """
        x_centered = x - self.b_dec
        pre_acts = F.linear(x_centered, self.W_enc, self.b_enc)
        return F.relu(pre_acts)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Args:
            f: Sparse features of shape (..., d_sae)

        Returns:
            Reconstructed activations of shape (..., d_model)
        """
        # W_dec has shape (d_model, d_sae)
        # F.linear expects weight of shape (out_features, in_features)
        # and computes input @ weight.T + bias
        # So we pass W_dec directly: f @ W_dec.T = (batch, d_sae) @ (d_sae, d_model)
        return F.linear(f, self.decoder_weights, self.b_dec)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: encode then decode.

        Args:
            x: Input activations of shape (..., d_model)
            return_features: If True, also return sparse features

        Returns:
            Reconstructed activations (and optionally sparse features)
        """
        features = self.encode(x)
        x_hat = self.decode(features)

        if return_features:
            return x_hat, features
        return x_hat

    def compute_loss(
        self, x: torch.Tensor, return_components: bool = False
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute SAE training loss: MSE reconstruction + L1 sparsity.

        Args:
            x: Input activations
            return_components: If True, return loss components dict

        Returns:
            Total loss (or dict with mse_loss, l1_loss, total_loss)
        """
        x_hat, features = self.forward(x, return_features=True)

        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(x_hat, x)

        # Sparsity loss (L1 on features)
        l1_loss = features.abs().mean()

        total_loss = mse_loss + self.config.l1_coefficient * l1_loss

        if return_components:
            return {
                "mse_loss": mse_loss,
                "l1_loss": l1_loss,
                "total_loss": total_loss,
                "l0_sparsity": (features > 0).float().sum(dim=-1).mean(),
            }
        return total_loss

    def get_feature_activations(
        self, x: torch.Tensor, top_k: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get active features and their indices.

        Args:
            x: Input activations of shape (..., d_model)
            top_k: If provided, return only top-k features per sample

        Returns:
            Tuple of (feature_values, feature_indices)
        """
        features = self.encode(x)

        if top_k is not None:
            values, indices = torch.topk(features, k=top_k, dim=-1)
            return values, indices

        # Return all non-zero features
        active_mask = features > 0
        return features, active_mask

    def save(self, path: Union[str, Path]):
        """Save SAE state dict and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "sae.pt")
        torch.save(
            {
                "d_model": self.config.d_model,
                "d_sae": self.config.d_sae,
                "l1_coefficient": self.config.l1_coefficient,
                "tied_weights": self.config.tied_weights,
                "normalize_decoder": self.config.normalize_decoder,
            },
            path / "config.pt",
        )

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "SparseAutoencoder":
        """Load SAE from saved state."""
        path = Path(path)
        config_dict = torch.load(path / "config.pt", map_location="cpu")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config = SAEConfig(
            d_model=config_dict["d_model"],
            d_sae=config_dict["d_sae"],
            l1_coefficient=config_dict["l1_coefficient"],
            tied_weights=config_dict["tied_weights"],
            normalize_decoder=config_dict["normalize_decoder"],
            device=device,
        )

        sae = cls(config)
        sae.load_state_dict(torch.load(path / "sae.pt", map_location=device))
        return sae


class SAETrainer:
    """Trainer for Sparse Autoencoders."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
    ):
        self.sae = sae
        self.optimizer = torch.optim.AdamW(
            sae.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.warmup_steps = warmup_steps
        self.step = 0

    def get_lr_multiplier(self) -> float:
        """Get learning rate multiplier for warmup."""
        if self.step < self.warmup_steps:
            return self.step / self.warmup_steps
        return 1.0

    def train_step(self, activations: torch.Tensor) -> dict[str, float]:
        """
        Single training step.

        Args:
            activations: Batch of model activations (batch_size, d_model)

        Returns:
            Dict of loss components
        """
        self.sae.train()

        # Adjust learning rate for warmup
        lr_mult = self.get_lr_multiplier()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_mult / max(lr_mult, 1e-8)

        self.optimizer.zero_grad()
        loss_dict = self.sae.compute_loss(activations, return_components=True)
        loss_dict["total_loss"].backward()
        self.optimizer.step()

        # Normalize decoder after update
        if self.sae.config.normalize_decoder:
            self.sae._normalize_decoder()

        self.step += 1

        return {k: v.item() for k, v in loss_dict.items()}

    def train(
        self,
        activation_store,  # Generator/iterable of activation batches
        num_steps: int,
        log_every: int = 100,
        save_every: int = 1000,
        save_path: Optional[str] = None,
    ) -> list[dict[str, float]]:
        """
        Full training loop.

        Args:
            activation_store: Iterable yielding activation batches
            num_steps: Number of training steps
            log_every: Log frequency
            save_every: Checkpoint frequency
            save_path: Path for saving checkpoints

        Returns:
            List of loss dictionaries
        """
        history = []

        pbar = tqdm(range(num_steps), desc="Training SAE")
        for step_idx in pbar:
            activations = next(iter(activation_store))

            if isinstance(activations, tuple):
                activations = activations[0]

            activations = activations.to(
                device=self.sae.config.device, dtype=self.sae.config.dtype
            )

            # Flatten if needed (batch, seq, d_model) -> (batch*seq, d_model)
            if activations.dim() == 3:
                activations = rearrange(activations, "b s d -> (b s) d")

            loss_dict = self.train_step(activations)
            history.append(loss_dict)

            if step_idx % log_every == 0:
                pbar.set_postfix(
                    mse=f"{loss_dict['mse_loss']:.4f}",
                    l1=f"{loss_dict['l1_loss']:.4f}",
                    l0=f"{loss_dict['l0_sparsity']:.1f}",
                )

            if save_path and (step_idx + 1) % save_every == 0:
                self.sae.save(f"{save_path}/step_{step_idx + 1}")

        return history
