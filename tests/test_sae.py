"""Tests for Sparse Autoencoder implementation."""

import pytest
import torch

from trace_lib.sae import SparseAutoencoder, SAEConfig, SAETrainer


class TestSAEConfig:
    def test_default_config(self):
        config = SAEConfig(d_model=768, d_sae=3072)
        assert config.d_model == 768
        assert config.d_sae == 3072
        assert config.l1_coefficient == 1e-3
        assert config.tied_weights is False
        assert config.normalize_decoder is True


class TestSparseAutoencoder:
    @pytest.fixture
    def sae(self):
        config = SAEConfig(
            d_model=64,
            d_sae=256,
            device="cpu",
            dtype=torch.float32,
        )
        return SparseAutoencoder(config)

    def test_init(self, sae):
        assert sae.W_enc.shape == (256, 64)
        assert sae.b_enc.shape == (256,)
        assert sae.W_dec.shape == (64, 256)
        assert sae.b_dec.shape == (64,)

    def test_encode(self, sae):
        x = torch.randn(8, 64)
        features = sae.encode(x)
        assert features.shape == (8, 256)
        # Features should be non-negative (ReLU)
        assert (features >= 0).all()

    def test_decode(self, sae):
        f = torch.relu(torch.randn(8, 256))
        x_hat = sae.decode(f)
        assert x_hat.shape == (8, 64)

    def test_forward(self, sae):
        x = torch.randn(8, 64)
        x_hat = sae(x)
        assert x_hat.shape == x.shape

    def test_forward_with_features(self, sae):
        x = torch.randn(8, 64)
        x_hat, features = sae(x, return_features=True)
        assert x_hat.shape == x.shape
        assert features.shape == (8, 256)

    def test_compute_loss(self, sae):
        x = torch.randn(8, 64)
        loss = sae.compute_loss(x)
        assert loss.ndim == 0  # Scalar
        assert loss > 0

    def test_compute_loss_components(self, sae):
        x = torch.randn(8, 64)
        loss_dict = sae.compute_loss(x, return_components=True)
        assert "mse_loss" in loss_dict
        assert "l1_loss" in loss_dict
        assert "total_loss" in loss_dict
        assert "l0_sparsity" in loss_dict

    def test_sparsity(self, sae):
        """Test that features are sparse (many are zero)."""
        x = torch.randn(100, 64)
        features = sae.encode(x)
        # Count non-zero features
        sparsity = (features == 0).float().mean()
        # We expect significant sparsity due to ReLU (at least ~40%)
        assert sparsity > 0.4

    def test_tied_weights(self):
        config = SAEConfig(
            d_model=64,
            d_sae=256,
            tied_weights=True,
            device="cpu",
        )
        sae = SparseAutoencoder(config)
        assert sae.W_dec is None
        # Decoder should be transpose of encoder
        assert torch.allclose(sae.decoder_weights, sae.W_enc.T)

    def test_get_feature_activations(self, sae):
        x = torch.randn(8, 64)
        values, indices = sae.get_feature_activations(x, top_k=10)
        assert values.shape == (8, 10)
        assert indices.shape == (8, 10)


class TestSAETrainer:
    @pytest.fixture
    def sae(self):
        config = SAEConfig(d_model=64, d_sae=256, device="cpu")
        return SparseAutoencoder(config)

    def test_train_step(self, sae):
        trainer = SAETrainer(sae)
        activations = torch.randn(32, 64)
        loss_dict = trainer.train_step(activations)
        assert "mse_loss" in loss_dict
        assert "l1_loss" in loss_dict
        assert trainer.step == 1

    def test_warmup(self, sae):
        trainer = SAETrainer(sae, warmup_steps=100)
        assert trainer.get_lr_multiplier() == 0.0
        trainer.step = 50
        assert trainer.get_lr_multiplier() == 0.5
        trainer.step = 100
        assert trainer.get_lr_multiplier() == 1.0
        trainer.step = 200
        assert trainer.get_lr_multiplier() == 1.0
