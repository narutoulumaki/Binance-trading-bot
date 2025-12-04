"""
Comprehensive Evaluation Script for YOLOv7
Computes mAP, FLOPs, FPS, latency, and model size metrics.
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict

from src.models import create_model
from src.dataset import create_dataloaders
from src.utils import (
    load_yaml, non_max_suppression, scale_coords,
    box_iou, xywh2xyxy, count_parameters
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv7')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/voc.yaml',
                        help='Path to dataset config')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output', type=str, default='results/metrics.json',
                        help='Output metrics file')
    return parser.parse_args()


def compute_ap(recall, precision):
    """Compute Average Precision (AP)."""
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def compute_map(pred_boxes, pred_scores, pred_labels, true_boxes, true_labels, 
                num_classes, iou_threshold=0.5):
    """Compute mean Average Precision (mAP)."""
    aps = []
    
    for c in range(num_classes):
        # Get predictions and ground truths for this class
        pred_mask = pred_labels == c
        true_mask = true_labels == c
        
        if not true_mask.any():
            continue
        
        class_pred_boxes = pred_boxes[pred_mask]
        class_pred_scores = pred_scores[pred_mask]
        class_true_boxes = true_boxes[true_mask]
        
        if len(class_pred_boxes) == 0:
            aps.append(0.0)
            continue
        
        # Sort by confidence
        sorted_indices = np.argsort(-class_pred_scores)
        class_pred_boxes = class_pred_boxes[sorted_indices]
        class_pred_scores = class_pred_scores[sorted_indices]
        
        # Match predictions to ground truths
        tp = np.zeros(len(class_pred_boxes))
        fp = np.zeros(len(class_pred_boxes))
        matched = set()
        
        for i, pred_box in enumerate(class_pred_boxes):
            best_iou = 0
            best_j = -1
            
            for j, true_box in enumerate(class_true_boxes):
                if j in matched:
                    continue
                
                iou = compute_iou_single(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_iou >= iou_threshold and best_j >= 0:
                tp[i] = 1
                matched.add(best_j)
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(class_true_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def compute_iou_single(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


@torch.no_grad()
def evaluate_model(model, dataloader, device, conf_thres, iou_thres, num_classes=20):
    """Evaluate model on dataset."""
    model.eval()
    
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_true_boxes = []
    all_true_labels = []
    
    # Latency tracking
    latencies = []
    
    print("\nEvaluating model...")
    for images, targets, _ in tqdm(dataloader):
        images = images.to(device)
        
        # Measure inference time
        start_time = time.time()
        predictions = model(images)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        latency = (time.time() - start_time) * 1000  # ms
        latencies.append(latency)
        
        # Post-process predictions
        predictions = non_max_suppression(
            predictions[0] if isinstance(predictions, tuple) else predictions,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        
        # Collect predictions and ground truths
        for i, pred in enumerate(predictions):
            # Predictions
            if len(pred):
                pred_boxes = pred[:, :4].cpu().numpy()
                pred_scores = pred[:, 4].cpu().numpy()
                pred_labels = pred[:, 5].cpu().numpy().astype(int)
                
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_pred_labels.append(pred_labels)
            else:
                all_pred_boxes.append(np.array([]))
                all_pred_scores.append(np.array([]))
                all_pred_labels.append(np.array([]))
            
            # Ground truths
            target_mask = targets[:, 0] == i
            if target_mask.any():
                target = targets[target_mask]
                true_labels = target[:, 1].cpu().numpy().astype(int)
                true_boxes = target[:, 2:6].cpu().numpy()
                # Convert xywh to xyxy
                true_boxes[:, 0] = (true_boxes[:, 0] - true_boxes[:, 2] / 2) * images.shape[3]
                true_boxes[:, 1] = (true_boxes[:, 1] - true_boxes[:, 3] / 2) * images.shape[2]
                true_boxes[:, 2] = (true_boxes[:, 0] + true_boxes[:, 2]) * images.shape[3]
                true_boxes[:, 3] = (true_boxes[:, 1] + true_boxes[:, 3]) * images.shape[2]
                
                all_true_boxes.append(true_boxes)
                all_true_labels.append(true_labels)
            else:
                all_true_boxes.append(np.array([]))
                all_true_labels.append(np.array([]))
    
    # Concatenate all predictions and ground truths
    pred_boxes = np.concatenate([b for b in all_pred_boxes if len(b)], axis=0) if any(len(b) for b in all_pred_boxes) else np.array([])
    pred_scores = np.concatenate([s for s in all_pred_scores if len(s)], axis=0) if any(len(s) for s in all_pred_scores) else np.array([])
    pred_labels = np.concatenate([l for l in all_pred_labels if len(l)], axis=0) if any(len(l) for l in all_pred_labels) else np.array([])
    true_boxes = np.concatenate([b for b in all_true_boxes if len(b)], axis=0) if any(len(b) for b in all_true_boxes) else np.array([])
    true_labels = np.concatenate([l for l in all_true_labels if len(l)], axis=0) if any(len(l) for l in all_true_labels) else np.array([])
    
    # Compute mAP@0.5
    map_50 = 0.0
    if len(pred_boxes) > 0 and len(true_boxes) > 0:
        map_50 = compute_map(pred_boxes, pred_scores, pred_labels, true_boxes, true_labels, num_classes, iou_threshold=0.5)
    
    # Compute latency statistics
    latencies = np.array(latencies)
    latency_p50 = np.percentile(latencies, 50)
    latency_p95 = np.percentile(latencies, 95)
    fps = 1000 / latency_p50
    
    return {
        'mAP@0.5': float(map_50),
        'latency_p50_ms': float(latency_p50),
        'latency_p95_ms': float(latency_p95),
        'fps': float(fps)
    }


def compute_flops(model, img_size=640):
    """Compute FLOPs for the model."""
    try:
        from thop import profile, clever_format
        input_tensor = torch.randn(1, 3, img_size, img_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except:
        return "N/A", "N/A"


def main():
    """Main evaluation function."""
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    cfg = checkpoint.get('config', {})
    
    # Create model
    model = create_model(cfg.get('model', 'yolov7_small'), nc=cfg.get('nc', 20))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Model parameters
    total_params, trainable_params = count_parameters(model)
    model_size_mb = total_params * 4 / (1024 * 1024)  # FP32
    
    # FLOPs
    flops, _ = compute_flops(model, args.img_size)
    
    # Load dataset
    data_cfg = load_yaml(args.data)
    _, val_loader = create_dataloaders(
        data_cfg,
        batch_size=1,
        img_size=args.img_size,
        workers=4
    )
    
    # Evaluate
    metrics = evaluate_model(
        model, val_loader, device,
        args.conf_thres, args.iou_thres,
        num_classes=cfg.get('nc', 20)
    )
    
    # Add model statistics
    metrics['total_parameters'] = int(total_params)
    metrics['trainable_parameters'] = int(trainable_params)
    metrics['model_size_mb'] = float(model_size_mb)
    metrics['flops'] = flops
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"Parameters: {metrics['total_parameters']:,}")
    print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
    print(f"FLOPs: {metrics['flops']}")
    print(f"Latency P50: {metrics['latency_p50_ms']:.2f} ms")
    print(f"Latency P95: {metrics['latency_p95_ms']:.2f} ms")
    print(f"FPS: {metrics['fps']:.1f}")
    print("="*70)
    
    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to: {output_path}")


if __name__ == '__main__':
    main()
