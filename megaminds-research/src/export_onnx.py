"""
Export YOLOv7 model to ONNX format for deployment.
"""

import torch
import onnx
import onnxsim
from pathlib import Path
import argparse

from src.models import create_model
from src.utils import load_yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export YOLOv7 to ONNX')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='models/yolov7.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model')
    return parser.parse_args()


def export_onnx(model, output_path, img_size=640, opset=13, simplify=True):
    """Export model to ONNX format."""
    print(f"\nExporting to ONNX...")
    print(f"  Input size: {img_size}x{img_size}")
    print(f"  Opset version: {opset}")
    
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Exported to: {output_path}")
    
    # Simplify
    if simplify:
        print("\nSimplifying ONNX model...")
        try:
            model_onnx = onnx.load(output_path)
            model_onnx, check = onnxsim.simplify(model_onnx)
            onnx.save(model_onnx, output_path)
            print("✅ Simplified successfully")
        except Exception as e:
            print(f"⚠️ Simplification failed: {e}")
    
    # Verify
    print("\nVerifying ONNX model...")
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print("✅ ONNX model verified")
    
    # Print model info
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n📊 Model size: {file_size:.2f} MB")


def main():
    """Main export function."""
    args = parse_args()
    
    print(f"Loading model from: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location='cpu')
    cfg = checkpoint.get('config', {})
    
    # Create model
    model = create_model(cfg.get('model', 'yolov7_small'), nc=cfg.get('nc', 20))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    export_onnx(
        model,
        str(output_path),
        img_size=args.img_size,
        opset=args.opset,
        simplify=args.simplify
    )
    
    print(f"\n✅ Export complete!")
    print(f"📁 ONNX model: {output_path}")


if __name__ == '__main__':
    main()
