import os
import json
import logging
from data_utils import load_dataset_sst2
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def quick_eval(model_path, test_loader, device):
    """
    Evaluate the float32-based model on disk with from_pretrained(...).
    Returns a naive param-based size (float32) + test accuracy.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)
    model.eval()
    
    size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)  # float32 assumption
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total else 0
    return size_mb, acc

if __name__ == "__main__":
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        
        # Load test data
        _, test_loader, _ = load_dataset_sst2(batch_size=8)
        
        # 1) Evaluate baseline
        baseline_metrics_path = "data/baseline_outputs/baseline_metrics.json"
        with open(baseline_metrics_path, "r") as f:
            base_metrics = json.load(f)
        logging.info(f"Baseline (trained) => size: {base_metrics['model_size_mb']:.2f} MB, accuracy: {base_metrics['accuracy']:.3f}")

        # 2) Evaluate Distilled
        dist_metrics_path = "data/distilled_model/distilled_metrics.json"
        with open(dist_metrics_path, "r") as f:
            dist_metrics = json.load(f)
        logging.info(f"Distilled => size: {dist_metrics['student_size_mb']:.2f} MB, accuracy: {dist_metrics['student_accuracy']:.3f}")

        # 3) Evaluate pruned/quant from metrics file
        pq_metrics_path = "data/pruned_quantized_model/pruning_quant_metrics.json"
        with open(pq_metrics_path, "r") as f:
            pq_metrics = json.load(f)
        
        # Our pruning script uses 'disk_size_mb' for the key
        pruned_size = pq_metrics["pruned_model"]["disk_size_mb"]
        pruned_acc  = pq_metrics["pruned_model"]["accuracy"]
        logging.info("Pruned => size on disk: %.2f MB, acc: %.3f" % (pruned_size, pruned_acc))

        pq_size = pq_metrics["pruned_quant_model"]["disk_size_mb"]
        pq_acc  = pq_metrics["pruned_quant_model"]["accuracy"]
        logging.info("Pruned+Quant => size on disk: %.2f MB, acc: %.3f" % (pq_size, pq_acc))

        # 4) Compare to baseline
        base_size = base_metrics["model_size_mb"]
        base_acc = base_metrics["accuracy"]

        # Distilled
        dist_size = dist_metrics["student_size_mb"]
        dist_acc  = dist_metrics["student_accuracy"]

        size_reduction_dist = (1 - (dist_size / base_size)) * 100
        acc_retention_dist  = (dist_acc / base_acc) * 100
        logging.info(f"[Distilled] Size reduction: {size_reduction_dist:.2f}%, Acc retention: {acc_retention_dist:.2f}%")

        # Pruned
        size_reduction_pruned = (1 - (pruned_size / base_size)) * 100
        acc_retention_pruned  = (pruned_acc / base_acc) * 100
        logging.info(f"[Pruned] Size reduction: {size_reduction_pruned:.2f}%, Acc retention: {acc_retention_pruned:.2f}%")

        # Pruned+Quant
        size_reduction_pq = (1 - (pq_size / base_size)) * 100
        acc_retention_pq  = (pq_acc / base_acc) * 100
        logging.info(f"[Pruned+Quant] Size reduction: {size_reduction_pq:.2f}%, Acc retention: {acc_retention_pq:.2f}%")

    except Exception as e:
        logging.error(f"Error in evaluate_compare script: {e}")
