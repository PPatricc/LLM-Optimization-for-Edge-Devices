import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from data_utils import load_dataset_sst2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def measure_accuracy(model, data_loader, device):
    model.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0

def magnitude_prune_once(model, prune_ratio):
    """
    Prune the smallest `prune_ratio` fraction of weights by magnitude across all layers.
    """
    all_weights = []
    # Gather all param tensors (excluding biases) into a single list
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            all_weights.extend(np.abs(param.data.cpu().numpy().flatten()))
    all_weights = np.array(all_weights)
    threshold = np.percentile(all_weights, prune_ratio * 100)

    # Now actually zero out weights below threshold
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            tensor = param.data.cpu().numpy()
            mask = np.abs(tensor) > threshold
            param.data = torch.from_numpy(tensor * mask).to(param.device)

def gradual_pruning(model, train_loader, device, steps=3, prune_per_step=0.1, epochs_per_step=1, lr=1e-5):
    """
    Example: do 3 pruning steps, each removing 10% additional weights,
    with some short finetuning between steps.

    steps=3 and prune_per_step=0.1 => total around 30% final, 
    but cumulative effect might be slightly more or less.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        # 1) short finetune
        model.train()
        for epoch in range(epochs_per_step):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        # 2) magnitude prune
        current_prune_ratio = (step+1) * prune_per_step
        logging.info(f"Gradual pruning step {step+1}/{steps}, total prune ratio ~{current_prune_ratio:.2f}")
        magnitude_prune_once(model, current_prune_ratio)
    return model

def dynamic_quantize(model):
    import torch.quantization
    logging.info("Applying dynamic quantization on linear layers.")
    q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return q_model

def get_size_on_disk(folder):
    total_bytes = 0
    for root, dirs, files in os.walk(folder):
        for filename in files:
            total_bytes += os.path.getsize(os.path.join(root, filename))
    return total_bytes / (1024**2)

if __name__ == "__main__":
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        _, test_loader, tokenizer = load_dataset_sst2(batch_size=8)

        # 1) Load the baseline model to be pruned
        baseline_path = "data/baseline_outputs/baseline_model"
        logging.info(f"Loading baseline model from {baseline_path}")
        model = AutoModelForSequenceClassification.from_pretrained(baseline_path, num_labels=2)

        # 2) Perform gradual pruning
        logging.info("Applying GRADUAL pruning...")
        model = gradual_pruning(model, train_loader=_, device=device, 
                                steps=3, prune_per_step=0.1, 
                                epochs_per_step=1, lr=1e-5)
        
        # Evaluate pruned
        pruned_acc = measure_accuracy(model, test_loader, device)
        pruned_dir = "data/pruned_model"
        os.makedirs(pruned_dir, exist_ok=True)
        model.save_pretrained(pruned_dir)
        tokenizer.save_pretrained(pruned_dir)
        pruned_disk_size = get_size_on_disk(pruned_dir)
        
        logging.info(f"[Gradual-Pruned] Disk size: {pruned_disk_size:.2f} MB, accuracy: {pruned_acc:.3f}")

        # 3) Dynamic quantization
        q_model = dynamic_quantize(model)
        quant_acc = measure_accuracy(q_model, test_loader, device)

        # Save quantized (the recommended way: state_dict + config)
        os.makedirs("data/pruned_quantized_model", exist_ok=True)
        torch.save(q_model.state_dict(), "data/pruned_quantized_model/quantized_model_state.pt")
        with open("data/pruned_quantized_model/config.json", "w") as f:
            f.write(model.config.to_json_string())

        pq_size = get_size_on_disk("data/pruned_quantized_model")
        logging.info(f"[Gradual-Pruned+Quant] Disk size: {pq_size:.2f} MB, accuracy: {quant_acc:.3f}")

        metrics = {
            "pruned_model": {
                "disk_size_mb": pruned_disk_size,
                "accuracy": pruned_acc
            },
            "pruned_quant_model": {
                "disk_size_mb": pq_size,
                "accuracy": quant_acc
            }
        }
        with open("data/pruned_quantized_model/pruning_quant_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        logging.error(f"Error in pruning_quantization script: {e}")
