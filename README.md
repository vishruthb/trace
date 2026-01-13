# trace

> **Mechanistic Interpretability under Precision Constraints**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

**trace** is a technical diagnostic framework designed to audit the structural integrity of AI safety mechanisms during model optimization. As models transition from high-precision research environments (FP32) to optimized production environments (INT8/INT4), the internal representations—or "circuits"—undergo non-linear transformations. **trace** programmatically identifies if these transformations lead to **"Safety Bit-Rot,"** where capability features are preserved but alignment-critical features are degraded or silenced.

## The Problem: The Systems-Safety Gap

In current ML workflows, quantization is treated as a performance optimization problem (minimizing Perplexity loss). However, from a safety perspective, even a small increase in Perplexity could represent the total collapse of a specific, sparsely-represented safety circuit.

**trace** addresses three primary risks:

1. **Feature Smearing:** Low-precision rounding causing distinct safety features to merge into a single noisy signal
2. **Activation Muting:** Safety-critical "refusal" neurons falling below the quantization clipping threshold
3. **Adversarial Divergence:** Quantization-induced noise creating new, "un-traced" pathways that adversarial prompts can exploit

## Installation

```bash
# Basic installation
pip install -e .

# Full installation with quantization support
pip install -e ".[full]"

# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
import trace_lib as trace

# Initialize Trace with a base and quantized model
audit = trace.Audit(
    base_model="meta-llama/Llama-3-8B",
    quant_model="meta-llama/Llama-3-8B-int8"
)

# Load pre-trained SAEs (or train your own)
audit.load_sae("/path/to/saes/")

# Trace specific safety circuits
results = audit.trace_features(dataset="harmful_refusal_v1")

# Generate report on feature integrity
print(results.summary())
# Output: "Warning: Feature #772 (Refusal) degraded by 22% in INT8"
```

> **Note:** The package is imported as `trace_lib` to avoid conflicts with Python's built-in `trace` module. You can alias it as `trace` for convenience.

## Core Concepts

### The Trace Metric: Cross-Precision Feature Recovery (CPFR)

trace measures how well safety-critical features are preserved during quantization using the **CPFR** metric:

```
CPFR(f) = 1 - ||f_ref - f_q||₂ / (||f_ref||₂ + ε)
```

Where:
- `f_ref` = Features extracted from the reference (high-precision) model
- `f_q` = Features extracted from the quantized model
- A drop in CPFR indicates corruption of the model's safety circuits

### Safety Fingerprints

trace creates a "Safety Fingerprint" by identifying which Sparse Autoencoder (SAE) features correspond to safety-critical behaviors:

- **Refusal** circuits that reject harmful requests
- **Uncertainty** circuits that express epistemic humility
- **Safety warning** circuits that provide appropriate disclaimers

## Architecture

### Module 1: `HookManager`

Low-level utility using PyTorch forward hooks to intercept tensors from the residual stream. Handles de-quantization of tensors on-the-fly for parity comparison.

```python
from trace_lib import HookManager, HookConfig

config = HookConfig(layers=[0, 5, 10], hook_points=["resid_post"])
hook_manager = HookManager(model, config)

with hook_manager.capture():
    model(input_ids)

activations = hook_manager.get_activations("resid_post", layer=5)
```

### Module 2: `CircuitTracer`

Maps specific safety behaviors to their corresponding SAE features, creating a reusable safety fingerprint.

```python
from trace_lib import CircuitTracer

tracer = CircuitTracer(model, sae_dict, tokenizer=tokenizer)
fingerprint = tracer.create_fingerprint(
    safety_prompts={"refusal": refusal_prompts},
    baseline_prompts=normal_prompts,
)
fingerprint.save("safety_fingerprint.json")
```

### Module 3: `OptimizationAuditor`

Core engine that runs systematic sweeps across different optimization stacks:

- **Post-Training Quantization (PTQ):** FP16, BF16, INT8, INT4 (GPTQ, AWQ)
- **Hardware Kernels:** Comparing standard CUDA kernels vs. optimized implementations

```python
from trace_lib import OptimizationAuditor
from trace_lib.optimization_auditor import QuantizationType, QuantizationConfig

auditor = OptimizationAuditor(model, sae_dict, tokenizer)
results = auditor.run_sweep(
    test_prompts=safety_prompts,
    configs=[
        QuantizationConfig(QuantizationType.INT8_DYNAMIC),
        QuantizationConfig(QuantizationType.INT4_GPTQ, bits=4),
    ]
)
print(results.best_config(min_cpfr=0.9))
```

## Metrics

### CPFR (Cross-Precision Feature Recovery)

Primary metric for measuring overall feature preservation:

```python
from trace_lib.metrics import CPFR

cpfr = CPFR(degradation_threshold=0.8)
result = cpfr.compute(features_ref, features_quant)
print(f"CPFR Score: {result.score}")
print(f"Degraded features: {result.degraded_features}")
```

### Feature Drift

Measures how individual features shift in activation space:

```python
from trace_lib.metrics import FeatureDrift

drift = FeatureDrift()
result = drift.compute(features_ref, features_quant)
print(f"Mean drift: {result.mean_drift}")
```

### Feature Smearing

Detects when distinct features merge due to low precision:

```python
from trace_lib.metrics import FeatureSmearing

smearing = FeatureSmearing()
result = smearing.detect_smearing(features_ref, features_quant)
print(f"Smeared pairs: {result['num_smeared_pairs']}")
```

### Activation Muting

Identifies features that fall below quantization thresholds:

```python
from trace_lib.metrics import ActivationMuting

muting = ActivationMuting()
result = muting.detect_muting(activations_ref, activations_quant)
print(f"Muting rate: {result['muting_rate']:.2%}")
```

## Experimental Test Bench

| Phase | Task | Technical Output |
|-------|------|------------------|
| **I** | Baseline Mapping | Train/Load SAEs on layers 0-N to identify 100+ "Alignment Features" |
| **II** | Precision Sweep | Progressively quantize weights and activations from 32-bit to 4-bit |
| **III** | Drift Analysis | Plot the decay curve of "Safety Features" vs. "Capability Features" |
| **IV** | Red-Teaming | Attempt "Quantization-Jailbreaks"—prompts that fail in M_ref but succeed in M_q |

## Built-in Datasets

trace includes curated datasets for common safety behaviors:

```python
from trace_lib.datasets import get_dataset, list_datasets

# List available datasets
print(list_datasets())
# ['harmful_refusal_v1', 'uncertainty_v1', 'safety_warning_v1', 'safety_comprehensive_v1']

# Load a dataset
dataset = get_dataset("harmful_refusal_v1")
print(f"Safety prompts: {len(dataset.safety_prompts)}")
print(f"Baseline prompts: {len(dataset.baseline_prompts)}")
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_audit.py` - Simple audit of a quantized model
- `precision_sweep.py` - Comprehensive sweep across quantization methods

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.35+
- See `pyproject.toml` for full dependencies

## License

MIT License - See LICENSE for details.

## Citation

```bibtex
@software{trace2024,
  title = {trace: Mechanistic Interpretability under Precision Constraints},
  year = {2024},
  url = {https://github.com/trace-interp/trace}
}
```

## Contributing

Contributions are welcome! Please see our contributing guidelines for more details.
