"""
Utility functions for training, evaluation, and data processing.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file=None, level=logging.INFO):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def save_checkpoint(state: Dict, save_path: str, is_best: bool = False):
    """Save model checkpoint."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path)
    if is_best:
        best_path = str(Path(save_path).parent / 'best.pt')
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer=None, device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0.0)
    
    return epoch, best_metric


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_yaml(yaml_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, yaml_path: str):
    """Save dictionary to YAML file."""
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def compute_flops(model: nn.Module, input_size=(1, 3, 640, 640)):
    """
    Compute FLOPs for the model.
    Requires thop library: pip install thop
    """
    try:
        from thop import profile, clever_format
        input_tensor = torch.randn(input_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return "N/A", "N/A"


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes.
    Args:
        box1: Tensor of shape (N, 4) in format (x1, y1, x2, y2)
        box2: Tensor of shape (M, 4) in format (x1, y1, x2, y2)
    Returns:
        iou: Tensor of shape (N, M)
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # Intersection
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - 
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    
    # Union
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def xywh2xyxy(x):
    """Convert boxes from [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xyxy2xywh(x):
    """Convert boxes from [x1, y1, x2, y2] to [x_center, y_center, w, h]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Non-Maximum Suppression (NMS) on inference results.
    
    Args:
        prediction: Tensor of shape (batch_size, num_boxes, 5+nc)
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
    
    Returns:
        List of detections per image [n, 6] (x1, y1, x2, y2, conf, cls)
    """
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Sort by confidence
        x = x[x[:, 4].argsort(descending=True)]

        # NMS
        c = x[:, 5:6] * 7680
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    
    return coords


def colorstr(*input):
    """Colors a string for terminal output."""
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


if __name__ == '__main__':
    print("Testing utility functions...")
    
    # Test seed
    set_seed(42)
    print("✅ Random seed set")
    
    # Test logger
    logger = setup_logger()
    logger.info("Logger initialized")
    print("✅ Logger working")
    
    # Test NMS
    pred = torch.randn(1, 25200, 25)  # batch_size=1, boxes=25200, 5+20 classes
    output = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    print(f"✅ NMS output: {len(output)} images, {output[0].shape} detections")
    
    print("\n✅ All utility tests passed!")
