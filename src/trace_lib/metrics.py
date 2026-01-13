"""
Metrics for measuring feature drift across precision boundaries.

Implements Cross-Precision Feature Recovery (CPFR) and related metrics
for quantifying safety circuit degradation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange


@dataclass
class CPFRResult:
    """Results from Cross-Precision Feature Recovery analysis."""

    # Overall CPFR score (0-1, higher is better)
    score: float

    # Per-feature recovery scores
    feature_scores: torch.Tensor

    # Features that degraded significantly
    degraded_features: list[int] = field(default_factory=list)

    # Features that were completely lost (recovery < threshold)
    lost_features: list[int] = field(default_factory=list)

    # Correlation between base and quantized features
    feature_correlations: Optional[torch.Tensor] = None

    # Metadata
    num_samples: int = 0
    degradation_threshold: float = 0.8

    def __repr__(self) -> str:
        return (
            f"CPFRResult(score={self.score:.4f}, "
            f"degraded={len(self.degraded_features)}, "
            f"lost={len(self.lost_features)})"
        )

    @property
    def integrity_status(self) -> str:
        """Human-readable integrity status."""
        if self.score > 0.95:
            return "EXCELLENT"
        elif self.score > 0.90:
            return "GOOD"
        elif self.score > 0.80:
            return "WARNING"
        elif self.score > 0.60:
            return "DEGRADED"
        else:
            return "CRITICAL"


class CPFR:
    """
    Cross-Precision Feature Recovery metric.

    Measures how well safety-critical features are preserved when
    transitioning from high-precision (M_ref) to quantized (M_q) models.

    The Trace Metric:
        CPFR(f) = 1 - ||f_ref - f_q||_2 / (||f_ref||_2 + ε)

    A drop in CPFR indicates that the model's safety circuits have been
    corrupted by the quantization process.
    """

    def __init__(
        self,
        degradation_threshold: float = 0.8,
        loss_threshold: float = 0.5,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        """
        Initialize CPFR metric.

        Args:
            degradation_threshold: Features below this score are "degraded"
            loss_threshold: Features below this score are "lost"
            normalize: Whether to normalize features before comparison
            eps: Small constant for numerical stability
        """
        self.degradation_threshold = degradation_threshold
        self.loss_threshold = loss_threshold
        self.normalize = normalize
        self.eps = eps

    def compute(
        self,
        features_ref: torch.Tensor,
        features_quant: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> CPFRResult:
        """
        Compute CPFR between reference and quantized features.

        Args:
            features_ref: Features from reference model (batch, d_sae) or (batch, seq, d_sae)
            features_quant: Features from quantized model (same shape)
            feature_mask: Optional mask for safety-critical features only

        Returns:
            CPFRResult with scores and degraded/lost feature analysis
        """
        # Flatten sequence dimension if present
        if features_ref.dim() == 3:
            features_ref = rearrange(features_ref, "b s d -> (b s) d")
            features_quant = rearrange(features_quant, "b s d -> (b s) d")

        # Ensure same device and dtype
        features_quant = features_quant.to(
            device=features_ref.device, dtype=features_ref.dtype
        )

        # Optional normalization
        if self.normalize:
            ref_norm = torch.norm(features_ref, dim=0, keepdim=True) + self.eps
            quant_norm = torch.norm(features_quant, dim=0, keepdim=True) + self.eps
            features_ref = features_ref / ref_norm
            features_quant = features_quant / quant_norm

        # Compute per-feature recovery scores
        # CPFR(f) = 1 - ||f_ref - f_q||_2 / (||f_ref||_2 + ε)
        ref_norms = torch.norm(features_ref, dim=0) + self.eps
        diff_norms = torch.norm(features_ref - features_quant, dim=0)

        feature_scores = 1.0 - (diff_norms / ref_norms)
        feature_scores = torch.clamp(feature_scores, min=0.0, max=1.0)

        # Apply mask if provided
        if feature_mask is not None:
            feature_scores = feature_scores * feature_mask

        # Compute correlation between features
        feature_correlations = self._compute_correlations(features_ref, features_quant)

        # Identify degraded and lost features
        degraded_features = (
            (feature_scores < self.degradation_threshold)
            .nonzero(as_tuple=True)[0]
            .tolist()
        )
        lost_features = (
            (feature_scores < self.loss_threshold).nonzero(as_tuple=True)[0].tolist()
        )

        # Overall CPFR score (weighted mean, weighting by feature activation magnitude)
        ref_activation_weights = torch.norm(features_ref, dim=0) + self.eps
        overall_score = (
            feature_scores * ref_activation_weights
        ).sum() / ref_activation_weights.sum()

        return CPFRResult(
            score=overall_score.item(),
            feature_scores=feature_scores.detach().cpu(),
            degraded_features=degraded_features,
            lost_features=lost_features,
            feature_correlations=feature_correlations.detach().cpu(),
            num_samples=features_ref.shape[0],
            degradation_threshold=self.degradation_threshold,
        )

    def _compute_correlations(
        self, features_ref: torch.Tensor, features_quant: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-feature Pearson correlations."""
        # Center features
        ref_centered = features_ref - features_ref.mean(dim=0, keepdim=True)
        quant_centered = features_quant - features_quant.mean(dim=0, keepdim=True)

        # Compute correlation
        ref_std = torch.std(ref_centered, dim=0) + self.eps
        quant_std = torch.std(quant_centered, dim=0) + self.eps

        correlations = (ref_centered * quant_centered).mean(dim=0) / (ref_std * quant_std)
        return correlations


@dataclass
class FeatureDriftResult:
    """Results from Feature Drift analysis."""

    # Mean drift magnitude across all features
    mean_drift: float

    # Maximum drift observed
    max_drift: float

    # Per-feature drift values
    feature_drifts: torch.Tensor

    # Features with drift above threshold
    high_drift_features: list[int] = field(default_factory=list)

    # Statistical summary
    drift_std: float = 0.0
    drift_median: float = 0.0

    def __repr__(self) -> str:
        return (
            f"FeatureDriftResult(mean={self.mean_drift:.4f}, "
            f"max={self.max_drift:.4f}, "
            f"high_drift={len(self.high_drift_features)})"
        )


class FeatureDrift:
    """
    Feature Drift metric for tracking representation changes.

    Measures how much individual features have shifted in activation
    space due to quantization or other transformations.

    Drift(f) = ||E[f_ref] - E[f_q]||_2 / σ(f_ref)

    Standardized by the reference feature standard deviation.
    """

    def __init__(self, drift_threshold: float = 2.0, eps: float = 1e-8):
        """
        Initialize Feature Drift metric.

        Args:
            drift_threshold: Features with drift > threshold are "high drift"
            eps: Small constant for numerical stability
        """
        self.drift_threshold = drift_threshold
        self.eps = eps

    def compute(
        self,
        features_ref: torch.Tensor,
        features_quant: torch.Tensor,
    ) -> FeatureDriftResult:
        """
        Compute feature drift between reference and quantized features.

        Args:
            features_ref: Features from reference model (batch, d_sae)
            features_quant: Features from quantized model (batch, d_sae)

        Returns:
            FeatureDriftResult with drift statistics
        """
        # Flatten if needed
        if features_ref.dim() == 3:
            features_ref = rearrange(features_ref, "b s d -> (b s) d")
            features_quant = rearrange(features_quant, "b s d -> (b s) d")

        # Compute mean activations
        mean_ref = features_ref.mean(dim=0)
        mean_quant = features_quant.mean(dim=0)

        # Compute reference standard deviation
        std_ref = features_ref.std(dim=0) + self.eps

        # Compute standardized drift per feature
        feature_drifts = torch.abs(mean_ref - mean_quant) / std_ref

        # Identify high drift features
        high_drift_features = (
            (feature_drifts > self.drift_threshold).nonzero(as_tuple=True)[0].tolist()
        )

        return FeatureDriftResult(
            mean_drift=feature_drifts.mean().item(),
            max_drift=feature_drifts.max().item(),
            feature_drifts=feature_drifts.detach().cpu(),
            high_drift_features=high_drift_features,
            drift_std=feature_drifts.std().item(),
            drift_median=feature_drifts.median().item(),
        )


class FeatureSmearing:
    """
    Detect Feature Smearing: when distinct safety features merge into noise.

    Smearing occurs when low-precision rounding causes multiple distinct
    features to activate together when they shouldn't.

    Measures increase in feature co-activation patterns.
    """

    def __init__(self, coactivation_threshold: float = 0.1):
        """
        Initialize Feature Smearing detector.

        Args:
            coactivation_threshold: Minimum activation for considering features "active"
        """
        self.coactivation_threshold = coactivation_threshold

    def compute_coactivation_matrix(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature co-activation matrix.

        Args:
            features: Feature activations (batch, d_sae)

        Returns:
            Co-activation matrix (d_sae, d_sae)
        """
        if features.dim() == 3:
            features = rearrange(features, "b s d -> (b s) d")

        # Binarize features
        active = (features > self.coactivation_threshold).float()

        # Compute co-activation (how often pairs of features fire together)
        coactivation = torch.mm(active.T, active) / features.shape[0]

        return coactivation

    def detect_smearing(
        self,
        features_ref: torch.Tensor,
        features_quant: torch.Tensor,
        smearing_threshold: float = 0.05,
    ) -> dict:
        """
        Detect smearing by comparing co-activation patterns.

        Args:
            features_ref: Reference features
            features_quant: Quantized features
            smearing_threshold: Increase in co-activation to flag as smearing

        Returns:
            Dict with smearing analysis
        """
        coact_ref = self.compute_coactivation_matrix(features_ref)
        coact_quant = self.compute_coactivation_matrix(features_quant)

        # Find increased co-activation (smearing)
        coact_increase = coact_quant - coact_ref

        # Mask diagonal (self-coactivation)
        mask = ~torch.eye(coact_increase.shape[0], dtype=torch.bool, device=coact_increase.device)
        coact_increase = coact_increase * mask

        # Find smeared feature pairs
        smeared_pairs = (coact_increase > smearing_threshold).nonzero(as_tuple=False)

        return {
            "num_smeared_pairs": smeared_pairs.shape[0] // 2,  # Symmetric, so divide by 2
            "smeared_pairs": smeared_pairs.tolist()[:100],  # Limit output
            "max_smearing": coact_increase.max().item(),
            "mean_smearing": coact_increase[mask].mean().item(),
            "coactivation_ref": coact_ref.detach().cpu(),
            "coactivation_quant": coact_quant.detach().cpu(),
        }


class ActivationMuting:
    """
    Detect Activation Muting: safety neurons falling below quantization threshold.

    When model weights/activations are quantized, neurons with small but
    meaningful activations can be rounded to zero, muting safety signals.
    """

    def __init__(self, muting_threshold: float = 0.01):
        """
        Initialize Activation Muting detector.

        Args:
            muting_threshold: Below this, activations are considered "muted"
        """
        self.muting_threshold = muting_threshold

    def detect_muting(
        self,
        activations_ref: torch.Tensor,
        activations_quant: torch.Tensor,
    ) -> dict:
        """
        Detect muted activations.

        Args:
            activations_ref: Reference activations
            activations_quant: Quantized activations

        Returns:
            Dict with muting analysis
        """
        if activations_ref.dim() == 3:
            activations_ref = rearrange(activations_ref, "b s d -> (b s) d")
            activations_quant = rearrange(activations_quant, "b s d -> (b s) d")

        # Features that were active in reference but muted in quantized
        active_ref = activations_ref.abs() > self.muting_threshold
        muted_quant = activations_quant.abs() <= self.muting_threshold

        # Muting events: was active, now muted
        muted = active_ref & muted_quant

        # Per-feature muting rate
        muting_rate = muted.float().mean(dim=0)

        # Features with significant muting
        significant_muting = (muting_rate > 0.1).nonzero(as_tuple=True)[0].tolist()

        return {
            "total_muting_events": muted.sum().item(),
            "muting_rate": muted.float().mean().item(),
            "features_with_muting": significant_muting,
            "per_feature_muting_rate": muting_rate.detach().cpu(),
            "max_feature_muting_rate": muting_rate.max().item(),
        }
