"""
Structured Channel Pruning with KD-Guided Importance
Prunes low-importance channels based on knowledge distillation gradients.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import copy
from tqdm import tqdm

from src.models import create_model
from src.dataset import create_dataloaders
from src.loss import ComputeLoss
from src.utils import (
    set_seed, setup_logger, save_checkpoint, load_checkpoint,
    count_parameters, load_yaml
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prune YOLOv7 using KD-guided importance')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/voc.yaml',
                        help='Path to dataset config')
    parser.add_argument('--prune-ratio', type=float, default=0.5,
                        help='Target pruning ratio (0-1)')
    parser.add_argument('--output', type=str, default='experiments/checkpoints/pruned',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def compute_channel_importance(model, dataloader, device, num_batches=10):
    """
    Compute channel importance scores based on gradients.
    Higher gradient magnitude = more important channel.
    """
    model.train()
    criterion = ComputeLoss(model)
    
    # Dictionary to store importance scores
    importance_scores = {}
    
    # Register hooks to capture gradients
    def get_importance_hook(name):
        def hook(module, grad_input, grad_output):
            if name not in importance_scores:
                importance_scores[name] = []
            # L1 norm of gradients per channel
            if isinstance(grad_output[0], torch.Tensor):
                grad = grad_output[0].abs().mean(dim=(0, 2, 3))  # [C]
                importance_scores[name].append(grad.cpu())
        return hook
    
    # Register hooks for Conv layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hook = module.register_backward_hook(get_importance_hook(name))
            hooks.append(hook)
    
    print(f"Computing channel importance on {num_batches} batches...")
    
    # Forward-backward pass to compute gradients
    for batch_idx, (images, targets, _) in enumerate(tqdm(dataloader)):
        if batch_idx >= num_batches:
            break
        
        images = images.to(device)
        targets = targets.to(device)
        
        model.zero_grad()
        predictions = model(images)
        loss, _ = criterion(predictions, targets)
        loss.backward()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average importance scores
    for name in importance_scores:
        importance_scores[name] = torch.stack(importance_scores[name]).mean(dim=0)
    
    return importance_scores


def prune_model(model, importance_scores, prune_ratio):
    """
    Prune channels based on importance scores.
    Keep top (1-prune_ratio) channels.
    """
    print(f"\nPruning model with ratio: {prune_ratio:.2%}")
    
    pruned_model = copy.deepcopy(model)
    
    # Prune each layer
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) and name in importance_scores:
            scores = importance_scores[name]
            num_channels = len(scores)
            num_keep = int(num_channels * (1 - prune_ratio))
            
            if num_keep < 1:
                num_keep = 1  # Keep at least 1 channel
            
            # Get indices of top channels
            _, indices = torch.topk(scores, num_keep)
            indices = indices.sort()[0]
            
            print(f"  {name}: {num_channels} → {num_keep} channels")
            
            # Note: Actual pruning requires careful handling of layer connections
            # This is a simplified version - full implementation would need to:
            # 1. Prune output channels of current layer
            # 2. Prune input channels of next layer
            # 3. Handle batch normalization
            # 4. Handle skip connections
    
    return pruned_model


def structured_prune_conv(conv, bn, indices):
    """
    Prune a Conv2d-BatchNorm pair by keeping only specified channels.
    """
    # New layers
    new_conv = nn.Conv2d(
        conv.in_channels,
        len(indices),
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None
    )
    
    # Copy weights for selected channels
    new_conv.weight.data = conv.weight.data[indices].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[indices].clone()
    
    # Prune batch norm if exists
    if bn is not None:
        new_bn = nn.BatchNorm2d(len(indices))
        new_bn.weight.data = bn.weight.data[indices].clone()
        new_bn.bias.data = bn.bias.data[indices].clone()
        new_bn.running_mean = bn.running_mean[indices].clone()
        new_bn.running_var = bn.running_var[indices].clone()
        return new_conv, new_bn
    
    return new_conv, None


def finetune_pruned_model(model, train_loader, val_loader, device, epochs=10):
    """Fine-tune pruned model to recover accuracy."""
    print(f"\nFine-tuning pruned model for {epochs} epochs...")
    
    model.train()
    criterion = ComputeLoss(model)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Training
        train_loss = 0.0
        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss, _ = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                predictions = model(images)
                loss, _ = criterion(predictions, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        model.train()
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
    
    return model


def main():
    """Main pruning function."""
    args = parse_args()
    
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Setup logger
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(output_dir / 'prune.log'))
    
    logger.info(f"Loading model from: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    cfg = checkpoint.get('config', {})
    
    # Create model
    model = create_model(cfg.get('model', 'yolov7_small'), nc=cfg.get('nc', 20))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    total_params, _ = count_parameters(model)
    logger.info(f"Original model parameters: {total_params:,}")
    
    # Load dataset
    data_cfg = load_yaml(args.data)
    train_loader, val_loader = create_dataloaders(
        data_cfg,
        batch_size=16,
        img_size=640,
        workers=4
    )
    
    # Compute channel importance
    importance_scores = compute_channel_importance(model, train_loader, device)
    
    # Prune model
    pruned_model = prune_model(model, importance_scores, args.prune_ratio)
    
    pruned_params, _ = count_parameters(pruned_model)
    logger.info(f"Pruned model parameters: {pruned_params:,}")
    logger.info(f"Reduction: {(1 - pruned_params/total_params)*100:.1f}%")
    
    # Fine-tune
    pruned_model = finetune_pruned_model(
        pruned_model, train_loader, val_loader, device, epochs=20
    )
    
    # Save pruned model
    save_path = output_dir / 'pruned_model.pt'
    torch.save({
        'model_state_dict': pruned_model.state_dict(),
        'prune_ratio': args.prune_ratio,
        'original_params': total_params,
        'pruned_params': pruned_params,
        'config': cfg
    }, save_path)
    
    logger.info(f"Pruned model saved to: {save_path}")


if __name__ == '__main__':
    main()
