# PD-YOLOv7: Pruned-Distilled YOLOv7 for Edge Deployment

## Research Project for Megaminds Internship (Round 1)

**Author**: Bharadhwaj K  
**Email**: bharadhwajdace@gmail.com  
**Submission Date**: TBD

---

## 🎯 Project Overview

This research implements **PD-YOLOv7**, a novel approach combining **Knowledge Distillation**, **Structured Channel Pruning**, and **Quantization-Aware Training** to optimize YOLOv7 for edge deployment while maintaining high detection accuracy.

### Key Contributions
1. **Knowledge Distillation Framework**: Teacher-student architecture with combined cross-entropy and KL divergence loss
2. **KD-Guided Pruning**: Structured channel pruning using knowledge distillation importance scores
3. **Quantization-Aware Training**: INT8 quantization for deployment on resource-constrained devices
4. **Comprehensive Evaluation**: Benchmarking on Pascal VOC 2007+2012 datasets

### Performance Targets
- **mAP@0.5**: ≥75% (aiming for <5% degradation from baseline)
- **Model Size**: <50% of original YOLOv7
- **FLOPs**: <60% of original computation
- **Latency**: >30 FPS on edge devices (NVIDIA Jetson/RPi with Coral TPU)

---

## 📁 Project Structure

```
megaminds-research/
├── data/                   # Dataset storage
│   ├── download_voc.py     # Pascal VOC download script
│   ├── voc.yaml            # Dataset config
│   └── VOCdevkit/          # Raw dataset
├── src/                    # Source code
│   ├── models.py           # YOLOv7 architecture
│   ├── train.py            # Baseline training
│   ├── train_kd.py         # KD training
│   ├── prune.py            # Structured pruning
│   ├── qat.py              # Quantization-aware training
│   ├── export_onnx.py      # ONNX export
│   └── eval.py             # Evaluation metrics
├── experiments/            # Training configs & logs
│   ├── configs/            # Experiment YAML files
│   ├── logs/               # TensorBoard logs
│   └── checkpoints/        # Saved models
├── results/                # Visualizations & metrics
│   ├── plots/              # Loss curves, mAP charts
│   ├── detections/         # Sample detection images
│   └── metrics.json        # Performance metrics
├── notebooks/              # Analysis notebooks
│   └── analysis.ipynb      # Results visualization
├── paper/                  # Research paper (LaTeX)
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── case_study/             # Business case study
│   └── case_study.md
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
cd data
python download_voc.py
```

### 3. Train Baseline Model
```bash
python src/train.py --config experiments/configs/baseline.yaml
```

### 4. Train with Knowledge Distillation
```bash
python src/train_kd.py --teacher checkpoints/yolov7-base.pt --config experiments/configs/kd.yaml
```

### 5. Apply Structured Pruning
```bash
python src/prune.py --model checkpoints/yolov7-kd.pt --prune-ratio 0.5
```

### 6. Quantization-Aware Training
```bash
python src/qat.py --model checkpoints/yolov7-pruned.pt
```

### 7. Export to ONNX
```bash
python src/export_onnx.py --model checkpoints/yolov7-qat.pt --opset 13
```

### 8. Evaluate Model
```bash
python src/eval.py --model checkpoints/yolov7-final.pt --data data/voc.yaml
```

---

## 📊 Methodology

### Stage 1: Baseline Training
- Train YOLOv7-Base (teacher) and YOLOv7-Small (student) on Pascal VOC
- Establish performance baselines (mAP, FLOPs, latency)

### Stage 2: Knowledge Distillation
- Implement soft target distillation from teacher to student
- Loss function: `L_total = α*CE_loss + β*KL_loss`
- Hyperparameters: α=0.7, β=0.3, temperature=3.0

### Stage 3: Structured Pruning
- Calculate channel importance using KD gradients
- Prune low-importance channels (target: 50% reduction)
- Fine-tune pruned model with KD

### Stage 4: Quantization-Aware Training
- Apply INT8 quantization to weights and activations
- QAT fine-tuning to recover accuracy
- Export to ONNX with quantization nodes

### Stage 5: Deployment Optimization
- Convert to TensorRT/ONNX Runtime
- Benchmark on edge devices
- Analyze latency distributions (P50, P95, P99)

---

## 📈 Expected Results

| Model               | mAP@0.5 | Params (M) | FLOPs (G) | FPS (RTX 3070) | Size (MB) |
|---------------------|---------|------------|-----------|----------------|-----------|
| YOLOv7-Base         | 82.3%   | 37.2       | 105.2     | 45             | 142       |
| YOLOv7-Small        | 78.1%   | 12.4       | 35.6      | 98             | 48        |
| PD-YOLOv7 (Ours)    | 76.5%   | 6.2        | 18.3      | 125            | 24        |
| **Improvement**     | -5.8%   | -83.3%     | -82.6%    | +177.8%        | -83.1%    |

---

## 📝 Deliverables

### 1. Research Paper
- **Title**: "PD-YOLOv7: Pruned-Distilled YOLOv7 for Efficient Edge Deployment"
- **Target Journals**: 
  - IEEE Access (Q2, IF: 3.476)
  - Applied Sciences (Q2, IF: 2.838)
  - Electronics (Q2, IF: 2.690)
  - Sensors (Q3, IF: 3.847)
  - Journal of Real-Time Image Processing (Q3, IF: 2.300)
- **Sections**: Abstract, Introduction, Literature Review, Proposed Method, Experiments, Results, Conclusion
- **References**: 25+ peer-reviewed papers with DOIs

### 2. Case Study
- **Problem Statement**: Edge device deployment challenges
- **Data Analysis**: Pascal VOC dataset characteristics
- **Model Development**: PD-YOLOv7 architecture
- **Key Insights**: Trade-offs between accuracy and efficiency
- **Business Recommendations**: Deployment scenarios (surveillance, robotics, IoT)

### 3. Video Presentation (15 minutes)
- Research novelty and motivation
- Implementation walkthrough
- Experimental results and visualizations
- Code demonstration
- Future work and limitations

### 4. Source Code
- Clean, documented Python codebase
- Reproducible experiments with seed control
- Pre-trained model checkpoints
- Inference scripts for edge devices

---

## 🔬 Research Questions

1. How does knowledge distillation improve pruning decisions compared to magnitude-based pruning?
2. What is the optimal balance between distillation loss and task loss (α, β hyperparameters)?
3. Can QAT recover accuracy lost during aggressive pruning (>50%)?
4. How does latency scale across different edge hardware (Jetson Nano, RPi 4, Coral TPU)?
5. What is the Pareto frontier for mAP vs. FLOPs trade-off?

---

## 📚 References (Preview)

1. Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M. (2022). *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors*. arXiv:2207.02696.
2. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network*. NIPS Deep Learning Workshop.
3. He, Y., et al. (2017). *Channel pruning for accelerating very deep neural networks*. ICCV.
4. Jacob, B., et al. (2018). *Quantization and training of neural networks for efficient integer-arithmetic-only inference*. CVPR.

*(Full bibliography in paper/references.bib)*

---

## 📧 Contact

**Bharadhwaj K**  
Email: bharadhwajdace@gmail.com  
GitHub: [narutoulumaki](https://github.com/narutoulumaki)

---

## 📄 License

This research is conducted as part of the Megaminds Internship Round 1 Assessment.  
Code will be made available under MIT License after submission.

---

*Last Updated: [Date]*
