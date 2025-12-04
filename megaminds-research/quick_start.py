"""
Quick start script - verify model and begin dataset download.
"""

import torch
from pathlib import Path
from src.models import yolov7_base, yolov7_small
from src.utils import count_parameters

print("\n" + "="*70)
print("🚀 YOLOv7 Research Project - Quick Start")
print("="*70)

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Create models
print("\n📦 Creating Models...")
print("\n   YOLOv7-Base (Teacher):")
model_base = yolov7_base(nc=20)
total, train = count_parameters(model_base)
print(f"      Parameters: {total:,}")
print(f"      Size reduction target: 50% → ~{total//2:,} params")

print("\n   YOLOv7-Small (Student):")
model_small = yolov7_small(nc=20)
total_s, train_s = count_parameters(model_small)
reduction = (1 - total_s / total) * 100
print(f"      Parameters: {total_s:,}")
print(f"      Reduction: {reduction:.1f}%")

# Test forward pass
print("\n🔄 Testing Forward Pass...")
x = torch.randn(1, 3, 640, 640)
model_base.eval()
with torch.no_grad():
    out = model_base(x)
print(f"   ✅ Input: {x.shape}")
print(f"   ✅ Output: {out[0].shape if isinstance(out, tuple) else out.shape}")

# Check dataset
print("\n📁 Dataset Status:")
voc_path = Path('data/VOCdevkit')
if voc_path.exists():
    print(f"   ✅ Dataset found at: {voc_path}")
    voc07 = voc_path / 'VOC2007'
    voc12 = voc_path / 'VOC2012'
    if voc07.exists():
        img_dir = voc07 / 'JPEGImages'
        if img_dir.exists():
            num_imgs = len(list(img_dir.glob('*.jpg')))
            print(f"   ✅ VOC2007: {num_imgs} images")
    if voc12.exists():
        img_dir = voc12 / 'JPEGImages'
        if img_dir.exists():
            num_imgs = len(list(img_dir.glob('*.jpg')))
            print(f"   ✅ VOC2012: {num_imgs} images")
else:
    print(f"   ❌ Dataset not found")
    print(f"   📥 Run: python data/download_voc.py")

print("\n" + "="*70)
print("✅ Setup Complete!")
print("="*70)

print("\n📝 Next Steps:")
print("   1. Download dataset (if not done):")
print("      python data/download_voc.py")
print("\n   2. Train baseline teacher model:")
print("      python src/train.py --config experiments/configs/baseline.yaml")
print("\n   3. Train student model:")
print("      python src/train.py --config experiments/configs/student.yaml")
print("\n   4. Train with knowledge distillation:")
print("      python src/train_kd.py --config experiments/configs/kd.yaml")

print("\n" + "="*70 + "\n")
