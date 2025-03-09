# **Domain-Specific Speech Recognition: Evaluation Report**

This short report summarizes the results of our **baseline model training**, **knowledge distillation**, and **iterative pruning + dynamic quantization** pipeline. The focus is on demonstrating common optimization techniques (distillation, pruning, quantization) and comparing accuracy vs. model size trade-offs.

---

## **1. Overview of Methods**

1. **Baseline Model (DistilBERT)**  
   - Loaded `distilbert-base-uncased` from Hugging Face and fine-tuned it on a **subset of 5,000** SST-2 training examples for **3 epochs**.
   - Achieved a higher accuracy than earlier quick demos due to longer training and more data than a trivial 1,000-sample slice.

2. **Knowledge Distillation**  
   - We used the trained baseline model (teacher) to train a smaller student model:
     - Student architecture: `google/bert_uncased_L-2_H-128_A-2`
     - Distillation hyperparameters: `temperature=4.0` and `alpha=0.3` (emphasizing teacher logits)
     - Trained for **5 epochs** on the same 5k data subset
   - Goal: reduce model size significantly while retaining a reasonable fraction of teacher accuracy.

3. **Gradual Pruning + Dynamic Quantization**  
   - Instead of pruning 30% in one shot, we **iteratively pruned** 10% at each step over three steps, with short re-training between steps. This often preserves more accuracy than one-shot pruning.
   - After pruning, we applied **PyTorch dynamic quantization** on linear layers, converting them to int8 for further size and speed benefits on CPU inference.

---

## **2. Results**

Below are final metrics. The table captures size (on disk, in MB), final accuracy on the validation set, and comparisons to the baseline:

| **Model**          | **Size (MB)** | **Accuracy** | **Comments**                              |
|--------------------|--------------:|-------------:|-------------------------------------------|
| **Baseline**       | 255.41        | 0.890        | 3 epochs, 5k training subset             |
| **Distilled**      | 16.73         | 0.768        | Tiny BERT (2 layers), 5 epochs distill   |
| **Pruned**         | 256.35        | 0.845        | \~30% weights removed (gradual)          |
| **Pruned+Quant**   | 132.29        | 0.833        | Additional int8 quantization             |

### **Relative Comparisons**

- **Distilled**  
  - Size Reduction: ~93% vs. baseline  
  - Accuracy Retention: ~86%  
  - Achieves a substantial model size decrease (from ~255MB to ~17MB).  

- **Gradual-Pruned (Float)**  
  - Negligible size difference on disk from the original float checkpoint (256MB vs. 255MB), likely because the final `.bin` still stores the same float parameters but with zeros.  
  - Maintains ~94.97% of baseline accuracy.  

- **Gradual-Pruned + Quant**  
  - Disk size: ~132MB (a 48% reduction from baseline).  
  - Accuracy: ~0.833 (about 93.56% of baseline).  

**Observations**:
- **Pruning** alone did not reduce the on-disk float checkpoint size significantly, but it can deliver real memory or compute savings if the zeroed weights are exploited at runtime.  
- **Dynamic Quantization** cut the final checkpoint size by roughly half again (from ~256MB to 132MB) while still preserving about 93–94% of baseline accuracy.

---

## **3. Potential Improvements**

1. **Full Training Dataset**  
   - We used only 5k samples from ~67k in SST-2 for faster experimentation. Using the entire dataset can significantly boost baseline accuracy (often 90–92% with DistilBERT), and similarly help the student and pruned/quantized models.

2. **Longer Training**  
   - While we did 3 epochs for the baseline and 5 epochs for the student, more extensive training or hyperparameter tuning could raise accuracy. Consider more epochs, a different learning rate, or a scheduler.

3. **Structured or Gradual Pruning**  
   - We already used gradual pruning in steps, which helps retain accuracy better than one-shot approaches. But we could further refine step sizes, do more steps, or adopt structured pruning (removing entire heads or filters) for real speedups.

4. **Quantization-Aware Training (QAT)**  
   - Instead of post-training dynamic quant, QAT can produce smaller models with better accuracy retention. This involves simulating quantization during the training process itself.

5. **Architecture Choices**  
   - The distilled student used here (L-2 H-128 A-2) is extremely small, giving a big size reduction but also a more pronounced accuracy drop. A bigger student or a DistilBERT-based student might strike a better balance.

6. **ONNX / TensorRT Exports**  
   - Deploying to edge devices often benefits from exporting the pruned+quantized model to ONNX or TensorRT to leverage hardware acceleration and further reduce latency.

---

## **4. Conclusions**

Overall, we see:
- **Baseline**: 0.89 accuracy with a 255MB model.  
- **Distilled**: ~93% smaller, retaining ~86% baseline accuracy.  
- **Pruned+Quant**: ~48% size reduction, ~93.6% baseline accuracy.  

These results illustrate **classic trade-offs** between compression, speed, and accuracy:

- Distillation yields significant size savings but a larger accuracy drop.  
- Pruning + quantization preserves more accuracy but yields a more moderate size reduction.  

In practical scenarios, further hyperparameter tuning, structured pruning, and quantization-aware training can refine these results to achieve higher accuracy or smaller footprints, depending on target constraints and user needs.
