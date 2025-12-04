"""
Quick test script to verify YOLOv7 setup.
Tests model creation, forward pass, and basic functionality.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import yolov7_base, yolov7_small
from src.utils import count_parameters, set_seed


def test_model_creation():
    """Test model creation and forward pass."""
    print("\n" + "="*70)
    print("🧪 Testing YOLOv7 Model Creation")
    print("="*70)
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test base model
    print("\n📦 Creating YOLOv7-Base (Teacher)...")
    model_base = yolov7_base(nc=20).to(device)
    total, trainable = count_parameters(model_base)
    print(f"   ✅ Parameters: {total:,} (trainable: {trainable:,})")
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    x = torch.randn(2, 3, 640, 640).to(device)
    model_base.eval()
    with torch.no_grad():
        output = model_base(x)
    
    if isinstance(output, tuple):
        print(f"   ✅ Output shape: {output[0].shape}")
        print(f"   ✅ Predictions shape: {output[1][0].shape}")
    else:
        print(f"   ✅ Output shape: {output.shape}")
    
    # Test small model
    print("\n📦 Creating YOLOv7-Small (Student)...")
    model_small = yolov7_small(nc=20).to(device)
    total, trainable = count_parameters(model_small)
    print(f"   ✅ Parameters: {total:,} (trainable: {trainable:,})")
    
    # Test forward pass
    model_small.eval()
    with torch.no_grad():
        output = model_small(x)
    
    if isinstance(output, tuple):
        print(f"   ✅ Output shape: {output[0].shape}")
    
    # Compare sizes
    params_base, _ = count_parameters(model_base)
    params_small, _ = count_parameters(model_small)
    reduction = (1 - params_small / params_base) * 100
    print(f"\n📊 Size comparison:")
    print(f"   Base model: {params_base:,} params")
    print(f"   Small model: {params_small:,} params")
    print(f"   Reduction: {reduction:.1f}%")
    
    print("\n✅ Model tests passed!")
    return True


def test_dataset():
    """Test dataset loading."""
    print("\n" + "="*70)
    print("🧪 Testing Dataset Loading")
    print("="*70)
    
    try:
        from src.dataset import VOCDataset
        
        # Check if dataset exists
        voc_path = Path('data/VOCdevkit')
        if not voc_path.exists():
            print("⚠️  VOC dataset not found at data/VOCdevkit")
            print("   Run: python data/download_voc.py")
            return False
        
        print("\n📦 Loading VOC2007 trainval...")
        dataset = VOCDataset(
            root_dir='data/VOCdevkit',
            year='2007',
            image_set='trainval',
            img_size=640,
            augment=False
        )
        print(f"   ✅ Dataset size: {len(dataset)} images")
        
        # Test loading one sample
        print("\n🔍 Loading sample...")
        image, targets, img_id = dataset[0]
        print(f"   ✅ Image shape: {image.shape}")
        print(f"   ✅ Targets shape: {targets.shape}")
        print(f"   ✅ Image ID: {img_id}")
        print(f"   ✅ Number of objects: {len(targets)}")
        
        print("\n✅ Dataset tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def test_loss():
    """Test loss computation."""
    print("\n" + "="*70)
    print("🧪 Testing Loss Computation")
    print("="*70)
    
    try:
        from src.models import yolov7_small
        from src.loss import ComputeLoss
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("\n📦 Creating model and loss function...")
        model = yolov7_small(nc=20).to(device)
        compute_loss = ComputeLoss(model)
        print("   ✅ Loss function created")
        
        # Dummy predictions and targets
        print("\n🔄 Computing loss on dummy data...")
        predictions = [
            torch.randn(2, 3, 80, 80, 25).to(device),
            torch.randn(2, 3, 40, 40, 25).to(device),
            torch.randn(2, 3, 20, 20, 25).to(device),
        ]
        
        targets = torch.tensor([
            [0, 5, 0.5, 0.5, 0.3, 0.3],
            [0, 10, 0.7, 0.3, 0.2, 0.4],
            [1, 15, 0.4, 0.6, 0.25, 0.35],
        ]).to(device)
        
        loss, loss_items = compute_loss(predictions, targets)
        
        print(f"   ✅ Total loss: {loss.item():.4f}")
        print(f"   ✅ Box loss: {loss_items[0]:.4f}")
        print(f"   ✅ Obj loss: {loss_items[1]:.4f}")
        print(f"   ✅ Cls loss: {loss_items[2]:.4f}")
        
        print("\n✅ Loss tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🚀 YOLOv7 Setup Verification")
    print("="*70)
    
    results = []
    
    # Test model
    results.append(("Model Creation", test_model_creation()))
    
    # Test dataset
    results.append(("Dataset Loading", test_dataset()))
    
    # Test loss
    results.append(("Loss Computation", test_loss()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to train.")
        print("\n📝 Next steps:")
        print("   1. Download dataset: python data/download_voc.py")
        print("   2. Train baseline: python src/train.py --config experiments/configs/baseline.yaml")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
