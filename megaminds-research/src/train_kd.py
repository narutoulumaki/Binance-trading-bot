"""
Knowledge Distillation Training Script for YOLOv7
Student learns from teacher model using soft targets and feature distillation.
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm

from src.models import yolov7_base, yolov7_small, create_model
from src.dataset import create_dataloaders
from src.loss import ComputeLoss, DistillationLoss
from src.utils import (
    set_seed, setup_logger, save_checkpoint, load_checkpoint,
    count_parameters, AverageMeter, EarlyStopping, load_yaml
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv7 with Knowledge Distillation')
    parser.add_argument('--config', type=str, default='experiments/configs/kd.yaml',
                        help='Path to KD config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def train_one_epoch_kd(student, teacher, train_loader, criterion, kd_criterion, 
                       optimizer, device, epoch, logger, cfg, writer):
    """Train for one epoch with knowledge distillation."""
    student.train()
    teacher.eval()
    
    loss_meter = AverageMeter()
    task_loss_meter = AverageMeter()
    kd_loss_meter = AverageMeter()
    
    alpha = cfg.get('alpha', 0.7)
    beta = cfg.get('beta', 0.3)
    temperature = cfg.get('temperature', 3.0)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [KD]")
    
    for batch_idx, (images, targets, _) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass - student
        student_preds = student(images)
        
        # Forward pass - teacher (no gradients)
        with torch.no_grad():
            teacher_preds = teacher(images)
        
        # Task loss (standard detection loss)
        task_loss, loss_items = criterion(student_preds, targets)
        
        # Distillation loss (KL divergence on outputs)
        kd_loss = torch.tensor(0.0, device=device)
        if isinstance(student_preds, tuple):
            student_out = student_preds[0]
            teacher_out = teacher_preds[0]
        else:
            student_out = student_preds
            teacher_out = teacher_preds
        
        # KL divergence on class predictions
        for i in range(len(student_preds) if isinstance(student_preds, (list, tuple)) else 1):
            if isinstance(student_preds, (list, tuple)):
                s_pred = student_preds[i]
                t_pred = teacher_preds[i]
            else:
                s_pred = student_out
                t_pred = teacher_out
            
            # Extract class logits (last nc channels)
            nc = cfg['nc']
            s_cls = s_pred[..., 5:5+nc]
            t_cls = t_pred[..., 5:5+nc]
            
            # Flatten for KL divergence
            s_cls_flat = s_cls.reshape(-1, nc) / temperature
            t_cls_flat = t_cls.reshape(-1, nc) / temperature
            
            kd_loss += torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(s_cls_flat, dim=-1),
                torch.nn.functional.softmax(t_cls_flat, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            if not isinstance(student_preds, (list, tuple)):
                break
        
        # Combined loss
        total_loss = alpha * task_loss + beta * kd_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update meters
        bs = images.size(0)
        loss_meter.update(total_loss.item(), bs)
        task_loss_meter.update(task_loss.item(), bs)
        kd_loss_meter.update(kd_loss.item(), bs)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'task': f'{task_loss_meter.avg:.4f}',
            'kd': f'{kd_loss_meter.avg:.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % cfg.get('log_period', 10) == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train_KD/total_loss', loss_meter.avg, global_step)
            writer.add_scalar('Train_KD/task_loss', task_loss_meter.avg, global_step)
            writer.add_scalar('Train_KD/kd_loss', kd_loss_meter.avg, global_step)
    
    logger.info(f"Epoch {epoch} [KD] - Total: {loss_meter.avg:.4f}, "
                f"Task: {task_loss_meter.avg:.4f}, KD: {kd_loss_meter.avg:.4f}")
    
    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, logger, writer):
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for images, targets, _ in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        predictions = model(images)
        loss, loss_items = criterion(predictions, targets)
        
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        
        pbar.set_postfix({'val_loss': f'{loss_meter.avg:.4f}'})
    
    logger.info(f"Validation - Loss: {loss_meter.avg:.4f}")
    writer.add_scalar('Val/loss', loss_meter.avg, epoch)
    
    return loss_meter.avg


def main():
    """Main KD training function."""
    args = parse_args()
    
    # Load config
    cfg = load_yaml(args.config)
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
    logger = setup_logger(log_file=str(save_dir / 'train_kd.log'))
    logger.info(f"KD Training config: {args.config}")
    logger.info(f"Device: {device}")
    
    # Create student model
    logger.info(f"Creating student model: {cfg['model']}")
    student = create_model(cfg['model'], nc=cfg['nc'])
    student = student.to(device)
    
    total_params, trainable_params = count_parameters(student)
    logger.info(f"Student parameters: {total_params:,}")
    
    # Load teacher model
    logger.info(f"Loading teacher model: {cfg['teacher_model']}")
    teacher = create_model(cfg['teacher_model'], nc=cfg['nc'])
    teacher = teacher.to(device)
    
    # Load teacher weights
    teacher_checkpoint = torch.load(cfg['teacher_weights'], map_location=device)
    teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher.eval()
    
    total_params, _ = count_parameters(teacher)
    logger.info(f"Teacher parameters: {total_params:,}")
    
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
    
    # Create loss functions
    criterion = ComputeLoss(student)
    kd_criterion = DistillationLoss(temperature=cfg.get('temperature', 3.0))
    
    # Create optimizer
    if cfg['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            student.parameters(),
            lr=cfg['lr0'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov=True
        )
    elif cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            student.parameters(),
            lr=cfg['lr0'],
            weight_decay=cfg['weight_decay']
        )
    
    # Create scheduler
    if cfg['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['epochs'],
            eta_min=cfg['lr0'] * cfg['lrf']
        )
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=cfg.get('patience', 30), mode='min')
    
    # Training loop
    logger.info("Starting KD training...")
    best_loss = float('inf')
    
    for epoch in range(1, cfg['epochs'] + 1):
        # Train
        train_loss = train_one_epoch_kd(
            student, teacher, train_loader, criterion, kd_criterion,
            optimizer, device, epoch, logger, cfg, writer
        )
        
        # Validate
        if epoch % cfg['val_period'] == 0:
            val_loss = validate(student, val_loader, criterion, device, epoch, logger, writer)
            
            # Check if best model
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                logger.info(f"New best KD model! Val loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
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
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': cfg
            }
            save_checkpoint(checkpoint, str(save_dir / f'epoch_{epoch}.pt'))
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Train_KD/lr', optimizer.param_groups[0]['lr'], epoch)
    
    logger.info("KD Training complete!")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
