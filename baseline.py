import os
import json
import logging
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_utils import load_dataset_sst2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def measure_performance(model, data_loader, device):
    """
    Return:
      - model_size_mb
      - avg_latency
      - accuracy
    """
    model.to(device)
    model.eval()

    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    
    latencies = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
            
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_latency = sum(latencies)/len(latencies)
    accuracy = correct/total if total>0 else 0.0

    return {
        "model_size_mb": model_size_mb,
        "avg_latency_seconds": avg_latency,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        train_loader, test_loader, tokenizer = load_dataset_sst2(batch_size=8, train_subset=5000, val_subset=1000)
        
        # Load or train baseline model
        model_name = "distilbert-base-uncased"
        logging.info(f"Loading base model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Train the model for 3 epochs (longer training)
        logging.info("Training the base model for 3 epochs on data...")
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()
        num_epochs = 3

        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate on test data
        logging.info("Measuring baseline performance...")
        baseline_metrics = measure_performance(model, test_loader, device)
        logging.info(f"Baseline metrics: {baseline_metrics}")

        # Save metrics
        os.makedirs("data/baseline_outputs", exist_ok=True)
        with open("data/baseline_outputs/baseline_metrics.json", "w") as f:
            json.dump(baseline_metrics, f, indent=2)
        
        # Save the trained baseline model
        model.save_pretrained("data/baseline_outputs/baseline_model")
        tokenizer.save_pretrained("data/baseline_outputs/baseline_model")
    
    except Exception as e:
        logging.error(f"Error in baseline script: {e}")
