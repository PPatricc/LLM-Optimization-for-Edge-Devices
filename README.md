# LLM Optimization for Edge Devices

This repository demonstrates how to:
1. Measure baseline performance of a pre-trained language model.
2. Implement knowledge distillation to train a smaller student model.
3. Apply magnitude-based pruning and post-training quantization.
4. Compare each optimized version to the original model in terms of size, latency, and accuracy.

## Setup

```bash
pip install torch transformers datasets scikit-learn sentencepiece
```

Quick Start
Gather or download dataset: (e.g. SST2 from Hugging Face)

### Quick Start

## Run baseline:

```bash
python baseline.py
```

This will train (or load) the base model, measure size, latency, and accuracy, and save metrics to baseline_metrics.json.

# Note
If you encounter such error:
2025-03-08 13:19:01,673 [ERROR] Error in baseline script: Invalid pattern: '**' can only be an entire path component
Please try to update the dataset by:

```bash
pip install --upgrade datasets huggingface_hub
```

## Distillation, Pruning, Quantization
Knowledge Distillation

```bash
python knowledge_distillation.py
```

- Loads the teacher (baseline) model.
- Trains a smaller student with chosen temperature & alpha.
- Saves the student metrics in data/distilled_model/distilled_metrics.json.


Pruning & Quantization

```bash
python pruning_quantization.py
```

- Loads the original model (baseline).
- Optionally applies magnitude-based pruning (gradual or one-shot).
- Applies dynamic int8 quantization to reduce model size on disk and memory usage.
- Saves final metrics to data/pruned_quantized_model/pruning_quant_metrics.json.

## Comparing Models

```bash
python evaluate_compare.py
```

Outputs and compares:

- Baseline (accuracy, size)
- Distilled (accuracy, size)
- Pruned / Pruned+Quant (accuracy, size)
…along with any relative percentage improvements (size reduction, accuracy retention).

## Potential Improvements

1. Train Longer & Use More Data
- Remove or reduce any .select(...) slicing in the dataset scripts for higher accuracy.
- Increase epochs in both baseline (baseline.py) and distillation (knowledge_distillation.py).

2. Hyperparameter Tuning
- Adjust distillation temperature, alpha, learning rate, batch size.
- Explore different pruning ratios or structured pruning techniques.

3. Quantization-Aware Training (QAT)
- Instead of post-training dynamic quant, do QAT to maintain better accuracy.
- Tools like bitsandbytes, Intel’s Neural Compressor, or OpenVINO can provide additional gains.

4. Structured / Gradual Pruning
- Removing entire attention heads or channels can yield better speedups.
- Gradual pruning with short re-training steps often retains more accuracy than one-shot pruning.

5. Real Edge Deployment
- Export final pruned/quantized model to ONNX or TensorRT.
- Test on actual resource-constrained hardware for accurate latency measurements.

6. Alternative Student Model
- A 2-layer mini-model might be too small for some tasks.
- Try DistilBERT or a 4-layer mini-BERT for a better balance of size & accuracy.
