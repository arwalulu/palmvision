# 🌴 PalmVision — Date Palm Leaf Disease Classification  
**EfficientNetB0 + CBAM (Attention-Enhanced CNN Pipeline)**  
**Deep Learning · Computer Vision · Precision Agriculture**

---

## 📌 Overview

PalmVision is an end-to-end deep learning pipeline for **automatic classification of date-palm leaf conditions**, distinguishing between:

- **Bug**
- **Dubas**
- **Healthy**
- **Honey**

The project implements a **fully reproducible, research-grade pipeline** including:

✔ Dataset cleaning & validation  
✔ EXIF normalization + deterministic preprocessing  
✔ Stratified dataset splitting  
✔ Training an **EfficientNetB0 + CBAM (Channel & Spatial Attention)** model  
✔ Full evaluation (confusion matrix, classification report, JSON metrics)  
✔ Organized experiments with versioned checkpoints  

Dataset Source (Mendeley Data):  
🔗 **https://data.mendeley.com/datasets/2nh364p2bc/2**

---

# 🧹 STEP 1 — Rigorous Preprocessing Pipeline

### ✅ 1. Raw dataset analysis
- 3000 raw candidate images scanned.
- Verified image integrity (no corrupt files).

### ✅ 2. Class-wise duplicate detection  
- In-class duplicates removed: **63**  
- Cross-class duplicates detected: **0**

### ✅ 3. EXIF correction & RGB normalization  
- All images corrected for orientation.  
- Converted to consistent RGB mode.  
- Saved to `data/normalized/`.

### **Final clean dataset counts**
| Class   | Count |
|---------|--------|
| Bug     | 541 |
| Dubas   | 797 |
| Healthy | 800 |
| Honey   | 799 |
| **Total** | **2937** |

---

# 🧪 STEP 2 — Stratified Splits (70/20/10)

| Split | Count | Bug | Dubas | Healthy | Honey |
|-------|--------|--------|--------|---------|--------|
| **Train (70%)** | 2055 | 378 | 558 | 560 | 559 |
| **Val (20%)**   | 588  | 109 | 159 | 160 | 160 |
| **Test (10%)**  | 294  | 54  | 80  | 80  | 80 |

✔ Perfect class balance preserved  
✔ Manifest files stored for reproducibility  

---

# 🧠 STEP 3 — EfficientNetB0 + CBAM Model

A hybrid model combining:

### **EfficientNetB0 (pretrained)**
- strong general visual features  
- frozen lower layers, fine-tuned upper blocks  

### **CBAM (Convolutional Block Attention Module)**
- **Channel Attention**  
- **Spatial Attention**  
- boosts discriminative focus on leaf texture patterns  

### Model Summary
Total params: 4.26M
Trainable params: 2.26M
Non-trainable: 2.00M
Output shape: (None, 4)

---

# 🎯 STEP 4 — Training

### Training settings:
- **Epochs:** 30 (+ Early Stopping)  
- **Optimizer:** Adam  
- **Learning Rate:** 3e-4 with ReduceLROnPlateau  
- **Batch Size:** 32  
- **Augmentation:** Horizontal Flip  

### Validation Results
Best epoch: **6**  
Validation accuracy: **~86.9%**

Artifacts saved automatically:
- Best model checkpoint  
- TensorBoard logs  
- Training history  

---

# 🧾 STEP 5 — Final Test Evaluation (Held-out, never seen before)

Test Accuracy: 86.73%
Test Loss: 0.4190

### Per-Class F1 Scores

| Class | F1-score |
|--------|------------|
| Bug     | 0.844 |
| Dubas   | 0.805 |
| Healthy | **0.981** |
| Honey   | 0.829 |

Healthy is easiest to classify; Dubas the hardest (expected in orchard datasets).

### Saved evaluation artifacts
- **Confusion Matrix**: `test_confusion_matrix.png`
- **Classification Report**: `test_classification_report.txt`
- **JSON Metrics**: `test_metrics.json`

---

# 📈 Why This Project Is AI-Specialist Level

🔹 Fully reproducible ML pipeline  
🔹 Correct dataset validation & cleaning  
🔹 EXIF-correct normalization (commonly overlooked)  
🔹 Stratified splitting done properly  
🔹 Custom EfficientNetB0 + CBAM architecture  
🔹 Professional callbacks (LR scheduler, checkpointing, early stopping)  
🔹 Test-only evaluation  
🔹 Organized experiment logging  
🔹 Modular code structure following best practices  

This setup mirrors real AI production pipelines.

---

# ▶️ Running the Project

### 1. Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-m1.txt
```
### 2. Build dataset splits
```bash
python -m src.data.build_splits
```
### 3. Train model
```bash
python -m src.training.train 
```

### 4. Evaluate best checkpoint on test set
```bash
python -m src.evaluation.eval_test
```





