import argparse
from io import BytesIO

import onnx
import torch
from ultralytics import YOLO
from models.common import optim

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PyTorch YOLOv8 to ONNX exporter")
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to PyTorch YOLOv8 weights')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--sim', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--input-shape', nargs='+', type=int, default=[1, 3, 640, 640], help='Model input shape for export')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for export')
    
    args = parser.parse_args()
    assert len(args.input_shape) == 4, "Input shape must be a list of 4 integers"
    return args


def load_model(weights_path, device):
    """Load, optimize, and move the model to the specified device"""
    yolov8 = YOLO(weights_path)
    model = yolov8.model.fuse().eval()
    
    for module in model.modules():
        optim(module)
        module.to(device)
    
    model.to(device)
    return model


def export_onnx(model, fake_input, save_path, opset_version):
    """Export the model to ONNX format"""
    with BytesIO() as buffer:
        torch.onnx.export(
            model,
            fake_input,
            buffer,
            opset_version=opset_version,
            input_names=['images'],
            output_names=['outputs', 'proto']
        )
        buffer.seek(0)
        onnx_model = onnx.load(buffer)
    
    onnx.checker.check_model(onnx_model)
    return onnx_model


def simplify_onnx_model(onnx_model):
    """Simplify the ONNX model"""
    try:
        simplified_model, check = onnxsim.simplify(onnx_model)
        assert check, 'Simplified ONNX model check failed'
        return simplified_model
    except Exception as e:
        print(f'Simplifier failure: {e}')
        return onnx_model


def main():
    args = parse_args()

    model = load_model(args.weights, args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)
    
    # Warm up the model by running it twice
    for _ in range(2):
        model(fake_input)
    
    save_path = args.weights.replace('.pt', '.onnx')
    onnx_model = export_onnx(model, fake_input, save_path, args.opset)
    
    if args.sim:
        onnx_model = simplify_onnx_model(onnx_model)
    
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')


if __name__ == '__main__':
    main()