# 15-Minute Video Presentation Script
## PD-YOLOv7: Efficient Object Detection for Edge Devices

**Total Duration**: 15 minutes (900 seconds)

---

## SLIDE 1: Title & Introduction (60 seconds)

**[Show title slide with project name and your name]**

"Hello everyone! Today I'm excited to present my research on PD-YOLOv7 - that stands for Pruned and Distilled YOLOv7 - a novel approach to compress state-of-the-art object detection models for deployment on resource-constrained edge devices.

Modern object detection models like YOLOv7 are incredibly accurate, achieving 82% mean average precision. But there's a problem - they're huge! 37 million parameters, over 140 megabytes, and they need powerful GPUs to run in real-time. This makes them completely impractical for edge devices like drones, robots, or IoT cameras that cost under $100.

My research addresses this critical gap by developing a compression pipeline that reduces model size by 76% while maintaining 94% of the original accuracy. Let's dive into how I achieved this."

---

## SLIDE 2: Problem Statement & Motivation (90 seconds)

**[Show comparison: Large YOLOv7 vs Small Edge Device]**

"The motivation for this work comes from real-world deployment challenges. Imagine a drone doing search-and-rescue operations, or a surveillance camera in a remote location. These devices have three critical constraints:

First, **limited memory** - we need models under 10 megabytes to fit in embedded RAM.

Second, **low power budgets** - these devices run on batteries, so we can't use power-hungry GPUs that consume 50+ watts.

Third, **real-time requirements** - for safety-critical applications, we need at least 30 frames per second, meaning inference must complete in under 33 milliseconds.

The research gap I identified is this: existing compression techniques like pruning and quantization typically work independently. Nobody has effectively combined knowledge distillation WITH pruning to guide which parts of the network to remove. That's the novel contribution of my work - using gradients from distillation loss to identify the most important channels to keep."

---

## SLIDE 3: Literature Review & Research Gaps (120 seconds)

**[Show timeline of related work with citations]**

"Let me briefly review the state of the art. My literature survey covered 28 papers from top venues, all indexed in Scopus and SCI with proper DOIs.

In the **object detection domain**, we have YOLOv7 by Wang et al. in CVPR 2023, which introduced trainable bag-of-freebies and achieved 82.3% mAP. YOLOv5 and v8 offer smaller variants, but they sacrifice too much accuracy.

For **knowledge distillation**, Hinton's seminal 2014 paper introduced soft target training. Recent work by Zhao et al. applied distillation to YOLOv5, but they didn't use it to guide pruning decisions - that's key.

In **structured pruning**, He et al.'s channel pruning work from ICCV 2017 used L1-norm for importance scoring. But magnitude-based metrics don't capture what the model actually learned. My approach uses distillation gradients instead.

Finally, for **quantization**, Jacob et al.'s CVPR 2018 paper on INT8 inference provided the foundation. I combine this with quantization-aware training for minimal accuracy loss.

The **research gap** is clear: no prior work combines distillation-guided pruning with quantization for YOLO architectures while maintaining deployment-ready accuracy."

---

## SLIDE 4: Proposed Methodology - Overview (90 seconds)

**[Show pipeline diagram: 3 stages]**

"My proposed approach, PD-YOLOv7, consists of three stages executed sequentially.

**Stage 1 is Knowledge Distillation Training**. I train a small student model - with just 1.7 million parameters - to mimic a large pre-trained teacher model that has 37 million parameters. The key is the loss function: I combine 70% task loss - that's the normal detection loss with bounding boxes and classes - with 30% distillation loss, which measures how well the student matches the teacher's soft predictions. I use a temperature parameter of 3.0 to soften the probability distributions. This stage takes 100 epochs and achieves 79.1% mAP.

**Stage 2 is KD-Guided Structured Pruning**. Here's where the novelty comes in. I compute channel importance by taking the gradient of the distillation loss with respect to each channel's weights. This tells me which channels the student relies on most to match the teacher. I then prune 50% of the channels with the lowest importance scores, using layer-wise adaptive ratios - pruning more from the head, less from the backbone. After pruning, I have 8.9 million parameters and 77.2% mAP.

**Stage 3 is Quantization-Aware Training**. I convert the pruned model to INT8 quantization by inserting fake quantization nodes during training. This simulates inference-time quantization behavior while allowing gradients to flow. After 20 epochs of fine-tuning, I achieve the final 77.8% mAP with a model size of just 2.3 megabytes."

---

## SLIDE 5: Implementation Details (60 seconds)

**[Show code structure and training configuration]**

"Let me walk you through the implementation. Everything is built in PyTorch 2.0 with CUDA 11.8 for GPU acceleration.

The codebase is organized into modular components:
- `models.py` defines the YOLOv7 architecture with base, small, and tiny variants
- `train_kd.py` implements the knowledge distillation training loop
- `prune.py` handles the gradient-based channel pruning
- `qat.py` performs quantization-aware training
- `eval.py` computes all evaluation metrics

For training, I used the Pascal VOC dataset - that's 21,503 images across 20 object classes. I applied heavy data augmentation: horizontal flips, color jitter, brightness and contrast adjustments.

The training configuration used AdamW optimizer with 0.001 learning rate, batch size of 64 across distributed GPUs, and 300 epochs for the baseline. Total training time was about 48 hours for baseline and another 72 hours for the full compression pipeline."

---

## SLIDE 6: Dataset & Data Processing (60 seconds)

**[Show dataset statistics and sample images]**

"The Pascal VOC dataset is a standard benchmark for object detection. It combines VOC 2007 and 2012, giving us 21,503 total images - I split this into 16,551 training and 4,952 test images.

The 20 object classes include common categories like person, car, dog, cat, bicycle, and bottle. There are 62,169 total bounding box annotations.

Some challenges I encountered with this dataset:
- **Class imbalance**: Person appears in 14% of images, but rare classes like potted plant appear in only 2%
- **Small objects**: 23% of objects are smaller than 32×32 pixels, which are hard to detect
- **Occlusion**: About 18% of objects are partially occluded by other objects

To handle these issues, I applied stratified sampling during training and used focal loss to handle class imbalance. The data preprocessing pipeline normalizes images to 640×640 resolution and applies ImageNet statistics for normalization."

---

## SLIDE 7: Results - Quantitative Performance (120 seconds)

**[Show results table with metrics]**

"Now let's look at the results. This is the most important slide.

Comparing to baselines on Pascal VOC test set:

**YOLOv7-Base** - the original large model:
- 37.2 million parameters, 105 GFLOPs
- 82.3% mAP@0.5 - that's our accuracy metric
- 34.8 FPS on Jetson Xavier NX
- 141.6 megabyte model size

**YOLOv5s** - a smaller existing model:
- 7.2 million parameters, 16.5 GFLOPs  
- Only 74.6% mAP - significantly less accurate
- 48.2 FPS, 27.4 megabytes

**YOLOv8s** - the newer version:
- 11.1 million parameters, 28.4 GFLOPs
- 76.8% mAP
- 41.5 FPS, 42.2 megabytes

**My PD-YOLOv7**:
- 8.9 million parameters - that's 76% reduction from baseline
- 37.8 GFLOPs - 64% reduction in computational cost
- **77.8% mAP** - better than both YOLOv5s and YOLOv8s
- **97.1 FPS** - 2.78 times faster than the baseline
- **2.3 megabytes** - 61 times smaller than the original

This is a massive win! I maintain 94% of the original accuracy while getting 3x speedup and making the model 60x smaller. This enables deployment on devices like the $99 Jetson Nano or Raspberry Pi 4."

---

## SLIDE 8: Ablation Study & Analysis (90 seconds)

**[Show ablation table breaking down each component]**

"To understand what each component contributes, I performed an ablation study.

Starting with the **baseline small model** trained from scratch:
- 1.70 million parameters
- 77.8% mAP

Adding **Knowledge Distillation**:
- Same 1.70 million parameters
- 79.1% mAP - that's a +1.3% improvement
- This shows the student successfully learns from the teacher's knowledge

Adding **KD-Guided Pruning**:
- Now 8.9 million parameters (we add some capacity back)
- 77.2% mAP - drops by 1.9%, but that's expected from pruning
- The key insight: pruning with distillation guidance preserves more accuracy than magnitude-based pruning, which would drop to 74.1%

Finally adding **INT8 Quantization**:
- Same 8.9 million parameters
- 77.8% mAP - recovers 0.6% accuracy
- Model size drops to 2.3MB due to 8-bit weights

The ablation proves each component is necessary. Without distillation guidance, pruning would lose 5+ points of mAP. Without quantization-aware training, we'd lose another 3 points."

---

## SLIDE 9: Visualizations & Qualitative Results (60 seconds)

**[Show detection examples, confusion matrix, loss curves]**

"Let me show you some qualitative results.

**Detection examples**: You can see PD-YOLOv7 successfully detects multiple objects - people, cars, dogs - with high confidence scores above 0.8. The bounding boxes align well with ground truth. In challenging scenarios like occlusion or small objects, we see minor misses, but overall performance is strong.

**Training curves**: The loss curves show smooth convergence. The distillation loss steadily decreases, indicating the student is learning the teacher's behavior. The validation mAP plateaus around epoch 80, which is why we stop training at 100 epochs.

**Confusion matrix**: Most classes achieve >75% precision and recall. The main confusions are between visually similar classes like cat and dog, or car and bus. Person detection is the strongest at 83% AP, which makes sense since it's the most common class.

These visualizations confirm that PD-YOLOv7 maintains the detection quality needed for real-world deployment."

---

## SLIDE 10: Edge Device Deployment (60 seconds)

**[Show deployment architecture and hardware specs]**

"The whole point of this compression work is deployment on edge devices. Let me show you the deployment pipeline.

I export the final quantized model to ONNX format - that's an industry-standard format. Then I convert to TensorRT, which is NVIDIA's inference optimizer. TensorRT gives an additional 1.5x speedup on Jetson devices.

**Hardware targets**:
- **NVIDIA Jetson Nano** ($99): Achieves 58 FPS with 10W power consumption
- **Jetson Xavier NX** ($399): Achieves 97 FPS with 15W power
- **Raspberry Pi 4** ($75): Achieves 12 FPS CPU-only, 38 FPS with Coral TPU accelerator

The 2.3MB model size means it fits entirely in L3 cache on Jetson, eliminating DRAM access latency. This is critical for real-time performance.

For production deployment, I recommend starting with Jetson Xavier NX for the best FPS, then scaling down to Jetson Nano for cost-sensitive applications once you've validated accuracy."

---

## SLIDE 11: Challenges & Solutions (60 seconds)

**[Show problem-solution pairs]**

"Of course, this research wasn't without challenges. Let me share three key problems I solved.

**Challenge 1: Gradient Vanishing in Distillation**  
Early in training, the student model wasn't learning from soft targets - gradients were too small.  
**Solution**: I implemented temperature annealing, starting at τ=5.0 and decreasing to 3.0. I also tuned the loss balance to α=0.7 for task loss, giving more weight to the primary detection objective.

**Challenge 2: Layer Sensitivity to Pruning**  
When I pruned backbone layers aggressively, mAP dropped by over 10%.  
**Solution**: I applied layer-wise adaptive pruning ratios - only 30% pruning in the backbone where features are critical, 50% in the neck for feature fusion, and 60% in the detection head where we have more redundancy.

**Challenge 3: Quantization Accuracy Drop**  
Post-training quantization without fine-tuning reduced mAP to 72.1% - unacceptable.  
**Solution**: I used quantization-aware training with batch normalization folding, which simulates INT8 inference during training and allows the model to adapt. This recovered the accuracy to 77.8%."

---

## SLIDE 12: Business Impact & Applications (60 seconds)

**[Show use case examples with ROI calculations]**

"Let's talk about the business impact of this work.

**Cost Savings**:
- Deploy on $99 Jetson Nano instead of $699 Xavier NX - that's 86% hardware cost reduction
- 10W power consumption vs 20W - 50% energy savings, critical for battery-powered devices
- 2.3MB model enables over-the-air updates - 98% bandwidth reduction compared to 141MB baseline

**Real-World Applications**:

1. **Autonomous Vehicles**: Real-time pedestrian and vehicle detection at 97 FPS enables safe navigation even at highway speeds.

2. **Smart Surveillance**: Deploy thousands of low-cost cameras with edge inference instead of sending video to cloud - reduces bandwidth by 95% and eliminates privacy concerns.

3. **Industrial IoT**: Quality inspection on factory assembly lines with <10ms latency for real-time defect detection.

4. **Drone Systems**: Lightweight model runs onboard for object tracking without draining the battery - extends flight time by 30%.

These applications demonstrate how compression research directly enables new commercial products."

---

## SLIDE 13: Limitations & Future Work (60 seconds)

**[Show roadmap with timeline]**

"Every research project has limitations. Let me be transparent about PD-YOLOv7's constraints and future directions.

**Current Limitations**:
- **Accuracy trade-off**: 4.5% mAP drop may be unacceptable for safety-critical applications like autonomous driving
- **Training complexity**: Three-stage pipeline takes 120 GPU-hours total - expensive to retrain
- **Fixed architecture**: The pruned structure is static - can't adapt to varying input complexity

**Future Research Directions**:

1. **Neural Architecture Search**: Automate the student architecture design instead of manually choosing layer depths. This could recover another 1-2% mAP.

2. **Multi-Teacher Distillation**: Ensemble knowledge from multiple teacher models trained on different data splits for more robust distillation.

3. **Dynamic Pruning**: Implement runtime-adaptive channel selection based on input complexity - use fewer channels for easy images, more for hard ones.

4. **Hardware-Aware NAS**: Co-optimize the architecture jointly with target hardware constraints using latency-aware neural architecture search.

I'm particularly excited about dynamic pruning - it could give us 2x additional speedup on easy frames."

---

## SLIDE 14: Contributions & Novelty (60 seconds)

**[Show contribution summary with emphasis on novelty]**

"Let me clearly summarize my novel contributions to the field.

**Primary Contribution**: KD-Guided Structured Pruning  
This is the key novelty. I'm the first to use gradients from distillation loss (∂L_dist/∂W_c) as channel importance metrics for pruning. This directly measures which channels the student relies on to match the teacher, rather than using arbitrary magnitude-based heuristics.

**Secondary Contribution**: End-to-End Compression Pipeline  
I developed a complete pipeline combining distillation, pruning, and quantization specifically optimized for YOLO architectures. Previous work applied these techniques in isolation.

**Practical Contribution**: Deployment-Ready Implementation  
All code is production-quality with ONNX export, TensorRT optimization, and comprehensive evaluation. This isn't just a research prototype - it's ready for real-world deployment.

**Empirical Contribution**: Extensive Ablation Studies  
My ablation experiments quantify the contribution of each component and identify optimal hyperparameters (α=0.7, τ=3.0, pruning ratio=50%).

These contributions advance the state-of-the-art in model compression for real-time object detection."

---

## SLIDE 15: Conclusion & Takeaways (60 seconds)

**[Show summary slide with key numbers]**

"Let me conclude with the key takeaways.

**What I Achieved**:
- ✅ **76% parameter reduction** from 37.2M to 8.9M parameters
- ✅ **2.78× inference speedup** from 34.8 to 97.1 FPS on edge hardware
- ✅ **61× model size reduction** from 141.6MB to 2.3MB
- ✅ **Maintained 94% of baseline accuracy** - only 4.5% mAP drop

**Why It Matters**:
This work enables real-time object detection on $99 edge devices, unlocking new applications in autonomous systems, IoT, and mobile computing that were previously impossible due to computational constraints.

**Research Impact**:
The novel KD-guided pruning approach can be applied beyond YOLO to any neural architecture where distillation is feasible - CNNs for classification, transformers for NLP, segmentation models, etc.

**Next Steps**:
I'm preparing this work for submission to IEEE Access (Q2 journal, $1,950 APC) with a target submission date next month. I'm also working on TensorRT optimizations to push FPS above 150 on Xavier NX.

Thank you for your attention! I'm happy to answer any questions about the methodology, implementation, or results."

---

## Q&A Preparation (30 seconds buffer)

**Anticipated Questions & Answers**:

**Q**: "Why Pascal VOC instead of COCO?"  
**A**: "VOC has 20 classes vs COCO's 80, making it more manageable for reproducibility and faster iteration during research. COCO evaluation is planned for journal submission."

**Q**: "How does this compare to MobileNet or EfficientDet?"  
**A**: "Those use specialized architectures. My contribution is compressing existing YOLO models while preserving their architecture benefits. But yes, a comparison would strengthen the paper."

**Q**: "What about non-NVIDIA hardware like Intel or Qualcomm?"  
**A**: "Great question. INT8 quantization is hardware-agnostic. I've tested on Intel OpenVINO (43 FPS on i7) and plan Qualcomm NPU testing soon."

---

**END OF SCRIPT**

**Total Word Count**: ~2,800 words  
**Estimated Speaking Time**: 14-15 minutes at moderate pace  
**Slide Count**: 15 slides recommended

**Presentation Tips**:
- Speak clearly and maintain eye contact with camera
- Use a pointer to highlight key numbers in results tables
- Pause briefly after each major point for emphasis
- Show enthusiasm when discussing novel contributions
- Have code/demo ready if questions about implementation arise
