"""
YOLOv7 Baseline Training Script
Train teacher and student models on Pascal VOC dataset.
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml
import sys

from src.models import yolov7_base, yolov7_small, yolov7_tiny
from src.dataset import create_dataloaders
from src.loss import ComputeLoss
from src.utils import (
    set_seed, setup_logger, save_checkpoint, load_checkpoint,
    count_parameters, AverageMeter, EarlyStopping, load_yaml
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv7 on Pascal VOC')
    parser.add_argument('--config', type=str, default='experiments/configs/baseline.yaml',
                        help='Path to training config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--data', type=str, default='data/voc.yaml',
                        help='Path to dataset config')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def create_model(model_name, nc=20):
    """Create model by name."""
    if model_name == 'yolov7_base':
        return yolov7_base(nc=nc)
    elif model_name == 'yolov7_small':
        return yolov7_small(nc=nc)
    elif model_name == 'yolov7_tiny':
        return yolov7_tiny(nc=nc)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger, cfg, writer):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    box_loss_meter = AverageMeter()
    obj_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
    
    for batch_idx, (images, targets, _) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        if cfg.get('amp', False):
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss, loss_items = criterion(predictions, targets)
        else:
            predictions = model(images)
            loss, loss_items = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        
        if cfg.get('amp', False):
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update meters
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        box_loss_meter.update(loss_items[0].item(), bs)
        obj_loss_meter.update(loss_items[1].item(), bs)
        cls_loss_meter.update(loss_items[2].item(), bs)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'box': f'{box_loss_meter.avg:.4f}',
            'obj': f'{obj_loss_meter.avg:.4f}',
            'cls': f'{cls_loss_meter.avg:.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % cfg.get('log_period', 10) == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/loss', loss_meter.avg, global_step)
            writer.add_scalar('Train/box_loss', box_loss_meter.avg, global_step)
            writer.add_scalar('Train/obj_loss', obj_loss_meter.avg, global_step)
            writer.add_scalar('Train/cls_loss', cls_loss_meter.avg, global_step)
    
    logger.info(f"Epoch {epoch} - Loss: {loss_meter.avg:.4f}, "
                f"Box: {box_loss_meter.avg:.4f}, Obj: {obj_loss_meter.avg:.4f}, Cls: {cls_loss_meter.avg:.4f}")
    
    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, logger, writer):
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter()
    box_loss_meter = AverageMeter()
    obj_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for images, targets, _ in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        predictions = model(images)
        loss, loss_items = criterion(predictions, targets)
        
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        box_loss_meter.update(loss_items[0].item(), bs)
        obj_loss_meter.update(loss_items[1].item(), bs)
        cls_loss_meter.update(loss_items[2].item(), bs)
        
        pbar.set_postfix({'val_loss': f'{loss_meter.avg:.4f}'})
    
    logger.info(f"Validation - Loss: {loss_meter.avg:.4f}, "
                f"Box: {box_loss_meter.avg:.4f}, Obj: {obj_loss_meter.avg:.4f}, Cls: {cls_loss_meter.avg:.4f}")
    
    # Log to tensorboard
    writer.add_scalar('Val/loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/box_loss', box_loss_meter.avg, epoch)
    writer.add_scalar('Val/obj_loss', obj_loss_meter.avg, epoch)
    writer.add_scalar('Val/cls_loss', cls_loss_meter.avg, epoch)
    
    return loss_meter.avg


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    cfg = load_yaml(args.config)
    if args.data:
        cfg['data'] = args.data
    if args.device:
        cfg['device'] = args.device
    
    # Setup
    set_seed(cfg.get('seed', 42))
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    save_dir = Path(cfg['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(log_file=str(save_dir / 'train.log'))
    logger.info(f"Training config: {args.config}")
    logger.info(f"Device: {device}")
    
    # Create model
    logger.info(f"Creating model: {cfg['model']}")
    model = create_model(cfg['model'], nc=cfg['nc'])
    model = model.to(device)
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Load dataset config
    data_cfg = load_yaml(cfg['data'])
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_cfg,
        batch_size=cfg['batch_size'],
        img_size=cfg['img_size'],
        workers=cfg['workers']
    )
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create loss function
    criterion = ComputeLoss(model)
    
    # Create optimizer
    if cfg['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg['lr0'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov=True
        )
    elif cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg['lr0'],
            weight_decay=cfg['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
    
    # Create scheduler
    if cfg['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['epochs'],
            eta_min=cfg['lr0'] * cfg['lrf']
        )
    elif cfg['scheduler'] == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=cfg['lrf'],
            total_iters=cfg['epochs']
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg['scheduler']}")
    
    # Resume from checkpoint
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch += 1
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=cfg.get('patience', 30), mode='min')
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, logger, cfg, writer
        )
        
        # Validate
        if epoch % cfg['val_period'] == 0:
            val_loss = validate(model, val_loader, criterion, device, epoch, logger, writer)
            
            # Check if best model
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                logger.info(f"New best model! Val loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': cfg
            }
            save_checkpoint(checkpoint, str(save_dir / f'epoch_{epoch}.pt'), is_best)
            
            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save periodic checkpoint
        if epoch % cfg.get('save_period', 10) == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': cfg
            }
            save_checkpoint(checkpoint, str(save_dir / f'epoch_{epoch}.pt'))
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
