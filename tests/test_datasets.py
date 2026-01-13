"""Tests for datasets module."""

import pytest

from trace_lib.datasets import (
    get_dataset,
    list_datasets,
    SafetyDataset,
    REFUSAL_SAFETY_PROMPTS,
    REFUSAL_BASELINE_PROMPTS,
)


class TestDatasets:
    def test_list_datasets(self):
        """Test that we can list available datasets."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "harmful_refusal_v1" in datasets

    def test_get_dataset(self):
        """Test loading a built-in dataset."""
        dataset = get_dataset("harmful_refusal_v1")
        assert isinstance(dataset, SafetyDataset)
        assert len(dataset.safety_prompts) > 0
        assert len(dataset.baseline_prompts) > 0
        assert "refusal" in dataset.behaviors

    def test_get_unknown_dataset(self):
        """Test that unknown datasets raise KeyError."""
        with pytest.raises(KeyError):
            get_dataset("nonexistent_dataset")

    def test_dataset_structure(self):
        """Test that datasets have proper structure."""
        for name in list_datasets():
            dataset = get_dataset(name)
            assert isinstance(dataset.name, str)
            assert isinstance(dataset.description, str)
            assert isinstance(dataset.safety_prompts, list)
            assert isinstance(dataset.baseline_prompts, list)
            assert isinstance(dataset.behaviors, list)
            assert len(dataset.safety_prompts) > 0
            assert len(dataset.baseline_prompts) > 0

    def test_refusal_prompts_content(self):
        """Test that refusal prompts contain expected content."""
        assert len(REFUSAL_SAFETY_PROMPTS) > 0
        assert len(REFUSAL_BASELINE_PROMPTS) > 0
        # Safety prompts should contain harmful-sounding content
        assert any("bomb" in p.lower() for p in REFUSAL_SAFETY_PROMPTS)
        # Baseline prompts should be benign
        assert any("capital" in p.lower() for p in REFUSAL_BASELINE_PROMPTS)

    def test_comprehensive_dataset(self):
        """Test that comprehensive dataset combines others."""
        comprehensive = get_dataset("safety_comprehensive_v1")
        refusal = get_dataset("harmful_refusal_v1")
        
        # Comprehensive should have more prompts
        assert len(comprehensive.safety_prompts) >= len(refusal.safety_prompts)
        # Should have multiple behaviors
        assert len(comprehensive.behaviors) > 1
