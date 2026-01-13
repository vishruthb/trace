"""
trace: Mechanistic Interpretability under Precision Constraints

A technical diagnostic framework designed to audit the structural integrity
of AI safety mechanisms during model optimization.

Example:
    import trace_lib as trace

    audit = trace.Audit(base_model="llama-3-8b", quant_model="llama-3-8b-int8")
    results = audit.trace_features(dataset="harmful_refusal_v1")
    print(results.summary())
"""

__version__ = "0.1.0"

from trace_lib.audit import Audit, FeatureIntegrityReport
from trace_lib.hook_manager import HookManager, HookConfig, DualHookManager
from trace_lib.circuit_tracer import CircuitTracer, SafetyFingerprint, SafetyFeature
from trace_lib.optimization_auditor import (
    OptimizationAuditor,
    QuantizationType,
    QuantizationConfig,
    AuditResult,
    SweepResult,
)
from trace_lib.metrics import (
    CPFR,
    CPFRResult,
    FeatureDrift,
    FeatureDriftResult,
    FeatureSmearing,
    ActivationMuting,
)
from trace_lib.sae import SparseAutoencoder, SAEConfig, SAETrainer
from trace_lib.datasets import get_dataset, list_datasets, SafetyDataset

__all__ = [
    # Main API
    "Audit",
    "FeatureIntegrityReport",
    # Hook Management
    "HookManager",
    "HookConfig",
    "DualHookManager",
    # Circuit Tracing
    "CircuitTracer",
    "SafetyFingerprint",
    "SafetyFeature",
    # Optimization Auditing
    "OptimizationAuditor",
    "QuantizationType",
    "QuantizationConfig",
    "AuditResult",
    "SweepResult",
    # Metrics
    "CPFR",
    "CPFRResult",
    "FeatureDrift",
    "FeatureDriftResult",
    "FeatureSmearing",
    "ActivationMuting",
    # SAE
    "SparseAutoencoder",
    "SAEConfig",
    "SAETrainer",
    # Datasets
    "get_dataset",
    "list_datasets",
    "SafetyDataset",
]
