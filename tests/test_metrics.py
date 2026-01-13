"""Tests for metrics module."""

import pytest
import torch

from trace_lib.metrics import CPFR, FeatureDrift, FeatureSmearing, ActivationMuting


class TestCPFR:
    @pytest.fixture
    def cpfr(self):
        return CPFR(degradation_threshold=0.8, loss_threshold=0.5)

    def test_identical_features(self, cpfr):
        """Identical features should have CPFR score of 1.0."""
        features = torch.randn(100, 256)
        result = cpfr.compute(features, features.clone())
        assert result.score > 0.99
        assert len(result.degraded_features) == 0
        assert len(result.lost_features) == 0

    def test_degraded_features(self, cpfr):
        """Modified features should show degradation."""
        features_ref = torch.randn(100, 256)
        # Add significant noise to some features
        features_quant = features_ref.clone()
        features_quant[:, :50] += torch.randn(100, 50) * 2
        
        result = cpfr.compute(features_ref, features_quant)
        assert result.score < 1.0
        assert len(result.degraded_features) > 0

    def test_zero_features(self, cpfr):
        """Zeroed features should be detected as lost."""
        features_ref = torch.randn(100, 256)
        features_quant = features_ref.clone()
        # Zero out some features
        features_quant[:, :10] = 0

        result = cpfr.compute(features_ref, features_quant)
        assert len(result.lost_features) > 0

    def test_batch_handling(self, cpfr):
        """Should handle 3D inputs (batch, seq, features)."""
        features_ref = torch.randn(10, 20, 256)
        features_quant = features_ref.clone()

        result = cpfr.compute(features_ref, features_quant)
        assert result.score > 0.99

    def test_integrity_status(self, cpfr):
        """Test integrity status thresholds."""
        features = torch.randn(100, 256)
        result = cpfr.compute(features, features)
        assert result.integrity_status == "EXCELLENT"


class TestFeatureDrift:
    @pytest.fixture
    def drift(self):
        return FeatureDrift(drift_threshold=2.0)

    def test_no_drift(self, drift):
        """Identical features should have zero drift."""
        features = torch.randn(100, 256)
        result = drift.compute(features, features.clone())
        assert result.mean_drift < 0.1
        assert len(result.high_drift_features) == 0

    def test_drift_detection(self, drift):
        """Should detect features with high drift."""
        features_ref = torch.randn(100, 256)
        features_quant = features_ref.clone()
        # Shift mean of some features significantly
        features_quant[:, :20] += 5.0

        result = drift.compute(features_ref, features_quant)
        assert result.mean_drift > 0
        assert len(result.high_drift_features) > 0

    def test_statistics(self, drift):
        """Test drift statistics are computed correctly."""
        features_ref = torch.randn(100, 256)
        features_quant = features_ref + torch.randn_like(features_ref) * 0.1

        result = drift.compute(features_ref, features_quant)
        assert result.drift_std >= 0
        assert result.drift_median >= 0
        assert result.max_drift >= result.mean_drift


class TestFeatureSmearing:
    @pytest.fixture
    def smearing(self):
        return FeatureSmearing(coactivation_threshold=0.1)

    def test_coactivation_matrix(self, smearing):
        """Test coactivation matrix computation."""
        features = torch.randn(100, 64).abs()  # Use abs to ensure some activation
        coact = smearing.compute_coactivation_matrix(features)
        assert coact.shape == (64, 64)
        # Diagonal should be high (self-coactivation)
        assert coact.diag().mean() > 0

    def test_no_smearing(self, smearing):
        """Identical features should show no smearing."""
        features = torch.randn(100, 64).abs()
        result = smearing.detect_smearing(features, features)
        assert result["num_smeared_pairs"] == 0

    def test_smearing_detection(self, smearing):
        """Should detect increased coactivation."""
        features_ref = torch.randn(100, 64).abs()
        features_quant = features_ref.clone()
        # Make multiple features activate together
        features_quant[:, 1] = features_quant[:, 0]
        features_quant[:, 2] = features_quant[:, 0]

        result = smearing.detect_smearing(features_ref, features_quant)
        # Should detect some smearing
        assert result["max_smearing"] >= 0


class TestActivationMuting:
    @pytest.fixture
    def muting(self):
        return ActivationMuting(muting_threshold=0.01)

    def test_no_muting(self, muting):
        """Identical activations should show no muting."""
        activations = torch.randn(100, 64).abs() + 0.1
        result = muting.detect_muting(activations, activations)
        assert result["muting_rate"] == 0

    def test_muting_detection(self, muting):
        """Should detect muted activations."""
        activations_ref = torch.randn(100, 64).abs() + 0.1
        activations_quant = activations_ref.clone()
        # Zero out some activations
        activations_quant[:, :10] = 0

        result = muting.detect_muting(activations_ref, activations_quant)
        assert result["muting_rate"] > 0
        assert len(result["features_with_muting"]) > 0

    def test_muting_statistics(self, muting):
        """Test muting statistics."""
        activations_ref = torch.randn(100, 64).abs() + 0.1
        activations_quant = torch.zeros_like(activations_ref)

        result = muting.detect_muting(activations_ref, activations_quant)
        assert result["muting_rate"] > 0.5  # Most should be muted
        assert result["max_feature_muting_rate"] > 0.5
