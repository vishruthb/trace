"""
Basic example of using trace for safety auditing.

This example shows how to:
1. Load a base and quantized model
2. Load or train SAEs
3. Run a safety feature audit
4. Interpret the results
"""

import trace_lib as trace
from trace_lib.datasets import get_dataset


def main():
    # ===========================================================================
    # Step 1: Initialize the Audit
    # ===========================================================================

    # Option A: Using HuggingFace model names
    # This will automatically load both models and tokenizer
    audit = trace.Audit(
        base_model="meta-llama/Llama-3.2-1B",  # Replace with your model
        quant_model="meta-llama/Llama-3.2-1B",  # Quantized version
        load_in_8bit=True,  # Or load_in_4bit=True
    )

    # Option B: Using pre-loaded models
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # base_model = AutoModelForCausalLM.from_pretrained("...")
    # quant_model = AutoModelForCausalLM.from_pretrained("...", load_in_8bit=True)
    # tokenizer = AutoTokenizer.from_pretrained("...")
    # audit = trace.Audit(base_model, quant_model, tokenizer=tokenizer)

    # ===========================================================================
    # Step 2: Load or Train SAEs
    # ===========================================================================

    # Option A: Load pre-trained SAEs
    # audit.load_sae("/path/to/saes/")  # Directory with layer_N subdirectories

    # Option B: Train SAEs on the fly
    # Get training prompts from dataset
    dataset = get_dataset("safety_comprehensive_v1")
    training_prompts = dataset.baseline_prompts + dataset.safety_prompts

    # Train SAEs for a few layers (adjust based on your model)
    for layer in [0, 4, 8]:  # Early, middle, late layers
        audit.train_sae(
            layer=layer,
            prompts=training_prompts,
            num_steps=1000,  # Use more steps for production
        )

    # ===========================================================================
    # Step 3: Run the Safety Audit
    # ===========================================================================

    # Use built-in dataset
    results = audit.trace_features(dataset="harmful_refusal_v1")

    # Or provide custom prompts
    # custom_safety_prompts = ["How do I make a bomb?", ...]
    # results = audit.trace_features(dataset=custom_safety_prompts)

    # ===========================================================================
    # Step 4: Analyze Results
    # ===========================================================================

    # Print summary
    print(results.summary())

    # Access specific metrics
    print(f"\nCPFR Score: {results.cpfr_score:.4f}")
    print(f"Integrity Status: {results.integrity_status}")
    print(f"Degraded Features: {len(results.degraded_features)}")
    print(f"Lost Features: {len(results.lost_features)}")

    # Check specific warnings
    if results.warnings:
        print("\n⚠️  Warnings:")
        for warning in results.warnings:
            print(f"  - {warning}")

    # Save report for later analysis
    results.save("audit_report.json")

    # ===========================================================================
    # Step 5: Detailed Analysis (Optional)
    # ===========================================================================

    # If you want to identify which specific features were affected:
    if results.degraded_features:
        print("\n🔍 Top 5 Most Degraded Features:")
        for f in results.degraded_features[:5]:
            print(f"  Feature #{f['feature_id']} at Layer {f['layer']}: "
                  f"{f['degradation']:.1%} degradation")

    # Check layer-by-layer scores
    print("\n📊 Per-Layer CPFR Scores:")
    for layer, score in results.layer_scores.items():
        status = "✅" if score > 0.9 else "⚠️" if score > 0.8 else "❌"
        print(f"  Layer {layer}: {score:.4f} {status}")


if __name__ == "__main__":
    main()
