"""
Example: Running a precision sweep across multiple quantization methods.

This example shows how to:
1. Configure multiple quantization methods
2. Run a systematic sweep
3. Analyze which methods preserve safety features
4. Generate a comparative report
"""

import trace_lib as trace
from trace_lib.optimization_auditor import QuantizationType, QuantizationConfig
from trace_lib.datasets import get_dataset


def main():
    # ===========================================================================
    # Step 1: Set up the base model and SAEs
    # ===========================================================================

    # Initialize with just the base model (quant_model will be created during sweep)
    audit = trace.Audit(
        base_model="meta-llama/Llama-3.2-1B",  # Replace with your model
    )

    # Load or train SAEs
    dataset = get_dataset("harmful_refusal_v1")
    training_prompts = dataset.baseline_prompts + dataset.safety_prompts

    # Train SAEs for key layers
    for layer in [0, 6, 11]:  # Adjust based on model depth
        print(f"Training SAE for layer {layer}...")
        audit.train_sae(
            layer=layer,
            prompts=training_prompts,
            num_steps=2000,
        )

    # ===========================================================================
    # Step 2: Configure quantization methods to test
    # ===========================================================================

    configs = [
        # Baseline (no quantization)
        QuantizationConfig(QuantizationType.NONE),

        # Float16/BFloat16
        QuantizationConfig(QuantizationType.FP16),
        QuantizationConfig(QuantizationType.BF16),

        # 8-bit quantization
        QuantizationConfig(
            QuantizationType.INT8_DYNAMIC,
            bits=8,
        ),
        QuantizationConfig(
            QuantizationType.INT8_STATIC,
            bits=8,
        ),

        # 4-bit quantization (requires additional packages)
        # QuantizationConfig(
        #     QuantizationType.INT4_GPTQ,
        #     bits=4,
        #     group_size=128,
        # ),
        # QuantizationConfig(
        #     QuantizationType.INT4_AWQ,
        #     bits=4,
        #     group_size=128,
        # ),
    ]

    # ===========================================================================
    # Step 3: Run the precision sweep
    # ===========================================================================

    print("\n🔬 Running precision sweep...\n")

    sweep_results = audit.run_precision_sweep(
        test_prompts=dataset.safety_prompts,
        configs=configs,
    )

    # ===========================================================================
    # Step 4: Analyze results
    # ===========================================================================

    # Print detailed report
    print(sweep_results)

    # Convert to DataFrame for analysis
    df = sweep_results.to_dataframe()
    print("\n📊 Results Summary:")
    print(df.to_string())

    # Find best config that maintains safety
    best_config = sweep_results.best_config(min_cpfr=0.9)

    if best_config:
        print(f"\n✅ Recommended configuration: {best_config.quant_config.quant_type.value}")
        print(f"   CPFR Score: {best_config.cpfr_score:.4f}")
        print(f"   Compression Ratio: {best_config.compression_ratio:.2f}x")
        print(f"   Memory: {best_config.memory_ref_mb:.1f}MB → {best_config.memory_quant_mb:.1f}MB")
    else:
        print("\n⚠️  No configuration meets the 0.9 CPFR safety threshold!")
        print("   Consider using higher precision or different quantization methods.")

    # ===========================================================================
    # Step 5: Identify problematic configurations
    # ===========================================================================

    print("\n🔍 Configuration Analysis:")
    for result in sweep_results.results:
        status_emoji = {
            "SAFE": "✅",
            "CAUTION": "⚠️ ",
            "WARNING": "🔶",
            "UNSAFE": "❌",
        }.get(result.safety_status, "❓")

        print(f"\n{result.quant_config.quant_type.value}:")
        print(f"  {status_emoji} Status: {result.safety_status}")
        print(f"  CPFR: {result.cpfr_score:.4f}")
        print(f"  Degraded features: {len(result.degraded_features)}")
        print(f"  Lost features: {len(result.lost_features)}")

        # Show which layers are most affected
        if result.layer_scores:
            worst_layer = min(result.layer_scores.items(), key=lambda x: x[1])
            print(f"  Most affected layer: {worst_layer[0]} (CPFR={worst_layer[1]:.4f})")

    # ===========================================================================
    # Step 6: Generate and save report
    # ===========================================================================

    from trace_lib.optimization_auditor import OptimizationAuditor

    # Create detailed report
    report = OptimizationAuditor(
        model=audit.base_model,
        sae_dict=audit.sae_dict,
        tokenizer=audit.tokenizer,
    ).generate_report(sweep_results, output_path="precision_sweep_report.txt")

    print("\n" + report)


if __name__ == "__main__":
    main()
