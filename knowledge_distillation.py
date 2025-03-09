import os
import logging
import torch
import torch.nn.functional as F
import json
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from data_utils import load_dataset_sst2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, device, temperature=4.0, alpha=0.3):
        """
        temperature=4.0 -> higher smoothness in teacher predictions
        alpha=0.3       -> more weight on soft distillation; 0.7 on real labels
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        # Hard loss: standard cross-entropy with ground-truth labels
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss: KL divergence between (student vs teacher) distributions
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Weighted sum
        # alpha=0.3 => 30% on hard labels, 70% on teacher knowledge
        return (1 - self.alpha) * hard_loss + (self.alpha) * soft_loss
    
    def train(self, train_loader, optimizer, epochs=5):
        """
        5-epoch training for the student. 
        If you have more compute, do 10 epochs or so.
        """
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                with torch.no_grad():
                    teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

                student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits

                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Distillation Loss: {avg_loss:.4f}")

def measure_model(model, data_loader, device):
    """ 
    Return size in MB & accuracy. 
    For a quick approximate measure. 
    """
    model.eval().to(device)
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
    
    correct, total = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total else 0
    return model_size_mb, accuracy

if __name__ == "__main__":
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        train_loader, test_loader, tokenizer = load_dataset_sst2(batch_size=8)

        # 1) Load teacher (baseline) model
        teacher_path = "data/baseline_outputs/baseline_model"
        logging.info(f"Loading teacher model from {teacher_path}")
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_path, num_labels=2)
        
        # 2) Initialize smaller student model
        # You can choose a bigger 'student' if you want better capacity:
        # e.g. "distilbert-base-uncased" or "google/bert_uncased_L-4_H-256_A-4"
        student_model_name = "google/bert_uncased_L-2_H-128_A-2"
        logging.info(f"Loading student model: {student_model_name}")
        student_model = AutoModelForSequenceClassification.from_pretrained(student_model_name, num_labels=2)
        
        # 3) Distillation
        dist_trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            device=device,
            temperature=4.0,   # bigger temperature
            alpha=0.3          # stronger emphasis on teacher logits
        )
        optimizer = AdamW(student_model.parameters(), lr=3e-5)
        dist_trainer.train(train_loader, optimizer, epochs=5)  # 5 epochs

        # 4) Evaluate & save
        student_size, student_acc = measure_model(student_model, test_loader, device)
        logging.info(f"Distilled student model size: {student_size:.2f} MB, accuracy: {student_acc:.3f}")

        os.makedirs("data/distilled_model", exist_ok=True)
        student_model.save_pretrained("data/distilled_model")
        tokenizer.save_pretrained("data/distilled_model")

        # Save metrics
        dist_metrics = {
            "student_size_mb": student_size,
            "student_accuracy": student_acc
        }
        with open("data/distilled_model/distilled_metrics.json", "w") as f:
            json.dump(dist_metrics, f, indent=2)
    
    except Exception as e:
        logging.error(f"Error in knowledge_distillation script: {e}")
