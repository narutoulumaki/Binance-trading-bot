"""
Quantization-Aware Training (QAT) for YOLOv7
Simulates INT8 quantization during training for deployment optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from pathlib import Path
import argparse
from tqdm import tqdm

from src.models import create_model
from src.dataset import create_dataloaders
from src.loss import ComputeLoss
from src.utils import (
    set_seed, setup_logger, save_checkpoint,
    count_parameters, load_yaml, AverageMeter
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantization-Aware Training for YOLOv7')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/voc.yaml',
                        help='Path to dataset config')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of QAT epochs')
    parser.add_argument('--output', type=str, default='experiments/checkpoints/qat',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


class QuantizableYOLOv7(nn.Module):
    """YOLOv7 wrapper for quantization."""
    
    def __init__(self, model):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        if isinstance(x, tuple):
            x = (self.dequant(x[0]), *x[1:])
        else:
            x = self.dequant(x)
        return x


def prepare_model_for_qat(model, device):
    """Prepare model for quantization-aware training."""
    print("Preparing model for QAT...")
    
    # Wrap model
    qat_model = QuantizableYOLOv7(model)
    qat_model.to(device)
    
    # Set quantization config
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare QAT
    qat_model = prepare_qat(qat_model, inplace=False)
    
    return qat_model


def train_qat(model, train_loader, val_loader, device, epochs, logger, output_dir):
    """Train model with quantization-aware training."""
    model.train()
    
    criterion = ComputeLoss(model.model)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Enable quantization after first few epochs
        if epoch > 3:
            model.apply(torch.quantization.enable_observer)
        
        # Training
        loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")
        for images, targets, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(images)
            loss, _ = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        logger.info(f"QAT Epoch {epoch}: Train Loss = {loss_meter.avg:.4f}")
        
        # Validation
        model.eval()
        val_loss_meter = AverageMeter()
        
        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                targets = targets.to(device)
                
                predictions = model(images)
                loss, _ = criterion(predictions, targets)
                
                val_loss_meter.update(loss.item(), images.size(0))
        
        val_loss = val_loss_meter.avg
        logger.info(f"QAT Epoch {epoch}: Val Loss = {val_loss:.4f}")
        
        model.train()
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = output_dir / 'best_qat.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, save_path)
            logger.info(f"Saved best QAT model: {val_loss:.4f}")
        
        scheduler.step()
    
    return model


def convert_to_quantized(model, device):
    """Convert QAT model to fully quantized model."""
    print("\nConverting to quantized model...")
    
    model.eval()
    model.cpu()
    
    # Convert to quantized
    quantized_model = convert(model, inplace=False)
    
    return quantized_model


def main():
    """Main QAT function."""
    args = parse_args()
    
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=str(output_dir / 'qat.log'))
    
    logger.info(f"Loading model from: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location='cpu')
    cfg = checkpoint.get('config', {})
    
    # Create model
    model = create_model(cfg.get('model', 'yolov7_small'), nc=cfg.get('nc', 20))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_params, _ = count_parameters(model)
    logger.info(f"Original model parameters: {total_params:,}")
    
    # Prepare for QAT
    qat_model = prepare_model_for_qat(model, device)
    
    # Load dataset
    data_cfg = load_yaml(args.data)
    train_loader, val_loader = create_dataloaders(
        data_cfg,
        batch_size=16,
        img_size=640,
        workers=4
    )
    
    # QAT training
    qat_model = train_qat(
        qat_model, train_loader, val_loader,
        device, args.epochs, logger, output_dir
    )
    
    # Convert to quantized
    quantized_model = convert_to_quantized(qat_model, device)
    
    # Save quantized model
    save_path = output_dir / 'quantized_model.pt'
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'config': cfg
    }, save_path)
    
    logger.info(f"Quantized model saved to: {save_path}")
    
    # Estimate size reduction
    original_size = total_params * 4 / (1024 * 1024)  # FP32 MB
    quantized_size = total_params * 1 / (1024 * 1024)  # INT8 MB
    logger.info(f"Size: {original_size:.2f}MB → {quantized_size:.2f}MB ({quantized_size/original_size*100:.1f}%)")


if __name__ == '__main__':
    main()
