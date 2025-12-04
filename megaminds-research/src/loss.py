"""
Loss functions for YOLOv7 training.
Implements ComputeLoss for box, objectness, and classification losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import box_iou


class ComputeLoss:
    """YOLOv7 loss computation."""
    
    def __init__(self, model, autobalance=False):
        """
        Args:
            model: YOLOv7 model
            autobalance: Whether to auto-balance loss components
        """
        device = next(model.parameters()).device
        h = model.detect
        
        # Get detection head info
        self.nc = h.nc  # number of classes
        self.nl = h.nl  # number of layers
        self.na = h.na  # number of anchors
        self.device = device
        self.autobalance = autobalance
        
        # Class weights
        self.cp = 1.0  # class positive weight
        self.cn = 0.0  # class negative weight
        
        # Loss weights
        self.balance = [4.0, 1.0, 0.4]  # P3, P4, P5
        self.box = 0.05
        self.obj = 1.0
        self.cls = 0.5
        
        # Build targets
        self.anchors = h.anchors
        self.stride = h.stride
        
        # BCE loss
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.cp], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.cp], device=device))
        
    def __call__(self, predictions, targets):
        """
        Compute loss.
        
        Args:
            predictions: List of 3 tensors [bs, na, h, w, no]
            targets: Tensor [num_targets, 6] (batch_idx, class, x, y, w, h)
        
        Returns:
            loss: Total loss
            loss_items: Tensor [box_loss, obj_loss, cls_loss]
        """
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        
        # Build targets for each layer
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)
        
        # Losses
        for i, pi in enumerate(predictions):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=self.device)
            
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                
                # Box loss
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = box_iou(pbox.T, tbox[i], x1y1x2y2=False)
                lbox += (1.0 - iou).mean()
                
                # Objectness
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
                
                # Classification
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=self.device)
                    cls_targets = tcls[i].long()
                    t[torch.arange(n), cls_targets] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)
            
            # Objectness loss
            lobj += self.BCEobj(pi[..., 4], tobj) * self.balance[i]
        
        # Scale losses
        lbox *= self.box
        lobj *= self.obj
        lcls *= self.cls
        
        bs = predictions[0].shape[0]
        loss = lbox + lobj + lcls
        
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()
    
    def build_targets(self, predictions, targets):
        """
        Build targets for each prediction layer.
        
        Args:
            predictions: List of 3 tensors
            targets: Tensor [num_targets, 6]
        
        Returns:
            tcls: List of class targets for each layer
            tbox: List of box targets for each layer
            indices: List of (b, a, gj, gi) indices
            anch: List of anchor tensors
        """
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        
        g = 0.5  # bias
        off = torch.tensor([
            [0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],
        ], device=targets.device).float() * g
        
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]
            
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Match by anchor aspect ratio
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < 4.0
                t = t[j]
                
                # Offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            
            # Append
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, int(gain[3]) - 1), gi.clamp_(0, int(gain[2]) - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        
        return tcls, tbox, indices, anch


class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss."""
    
    def __init__(self, temperature=3.0):
        """
        Args:
            temperature: Distillation temperature
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits):
        """
        Compute KL divergence loss for distillation.
        
        Args:
            student_logits: Student model outputs [bs, ...]
            teacher_logits: Teacher model outputs [bs, ...]
        
        Returns:
            loss: KL divergence loss
        """
        # Soften distributions
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence
        loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss


def box_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    """
    Calculate IoU between boxes.
    
    Args:
        box1: [N, 4]
        box2: [M, 4]
        x1y1x2y2: Whether boxes are in xyxy format
    
    Returns:
        iou: [N, M]
    """
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Convert from xywh to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # Intersection area
    inter = (torch.min(b1_x2[:, None], b2_x2) - torch.max(b1_x1[:, None], b2_x1)).clamp(0) * \
            (torch.min(b1_y2[:, None], b2_y2) - torch.max(b1_y1[:, None], b2_y1)).clamp(0)
    
    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1[:, None] * h1[:, None] + w2 * h2 - inter + eps
    
    iou = inter / union
    return iou


if __name__ == '__main__':
    print("Testing loss functions...")
    
    # Create dummy model
    from src.models import yolov7_small
    model = yolov7_small(nc=20)
    
    # Test ComputeLoss
    compute_loss = ComputeLoss(model)
    
    # Dummy predictions and targets
    predictions = [
        torch.randn(2, 3, 80, 80, 25),  # P3
        torch.randn(2, 3, 40, 40, 25),  # P4
        torch.randn(2, 3, 20, 20, 25),  # P5
    ]
    
    targets = torch.tensor([
        [0, 5, 0.5, 0.5, 0.3, 0.3],  # batch_idx, class, x, y, w, h
        [0, 10, 0.7, 0.3, 0.2, 0.4],
        [1, 15, 0.4, 0.6, 0.25, 0.35],
    ])
    
    loss, loss_items = compute_loss(predictions, targets)
    print(f"\n✅ Loss computed:")
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Box loss: {loss_items[0]:.4f}")
    print(f"   Obj loss: {loss_items[1]:.4f}")
    print(f"   Cls loss: {loss_items[2]:.4f}")
    
    # Test DistillationLoss
    distill_loss = DistillationLoss(temperature=3.0)
    student_logits = torch.randn(4, 20)
    teacher_logits = torch.randn(4, 20)
    kd_loss = distill_loss(student_logits, teacher_logits)
    print(f"\n✅ KD loss: {kd_loss.item():.4f}")
    
    print("\n✅ Loss function tests passed!")
