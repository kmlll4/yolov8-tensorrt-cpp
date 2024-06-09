import argparse
from io import BytesIO

import onnx
import torch
from ultralytics import YOLO
from models.common import PostDetect, optim

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 to ONNX Exporter")
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to PyTorch YOLOv8 weights')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS plugin')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for NMS plugin')
    parser.add_argument('--topk', type=int, default=100, help='Maximum number of detection bounding boxes')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--sim', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--input-shape', nargs='+', type=int, default=[1, 3, 640, 640], help='Model input shape')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for export')
    args = parser.parse_args()
    
    # Validate input shape length
    assert len(args.input_shape) == 4, "Input shape must be a list of 4 integers"
    
    # Set post-detection parameters
    PostDetect.conf_thres = args.conf_thres
    PostDetect.iou_thres = args.iou_thres
    PostDetect.topk = args.topk
    return args


def load_and_prepare_model(weights, device):
    yolov8 = YOLO(weights)
    model = yolov8.model.fuse().eval()
    
    # Optimize and move model to device
    for module in model.modules():
        optim(module)
        module.to(device)
    model.to(device)
    
    return model


def export_to_onnx(model, fake_input, save_path, opset_version):
    with BytesIO() as buffer:
        torch.onnx.export(
            model,
            fake_input,
            buffer,
            opset_version=opset_version,
            input_names=['images'],
            output_names=['num_dets', 'bboxes', 'scores', 'labels']
        )
        buffer.seek(0)
        onnx_model = onnx.load(buffer)
    
    onnx.checker.check_model(onnx_model)
    return onnx_model


def set_dynamic_shapes(onnx_model, batch_size, topk):
    shapes = [batch_size, 1, batch_size, topk, 4, batch_size, topk, batch_size, topk]
    for output in onnx_model.graph.output:
        for dim in output.type.tensor_type.shape.dim:
            dim.dim_param = str(shapes.pop(0))


def simplify_onnx_model(onnx_model):
    simplified_model, check = onnxsim.simplify(onnx_model)
    if not check:
        raise ValueError("Simplified ONNX model check failed")
    
    return simplified_model


def main(args):
    model = load_and_prepare_model(args.weights, args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)
    
    # Run model twice to ensure it's ready for export
    for _ in range(2):
        model(fake_input)
    
    save_path = args.weights.replace('.pt', '.onnx')
    onnx_model = export_to_onnx(model, fake_input, save_path, args.opset)
    set_dynamic_shapes(onnx_model, args.input_shape[0], args.topk)
    
    if args.sim:
        if onnxsim is None:
            print("onnxsim is not available. Please install onnxsim to use the --sim option.")
            return
        try:
            onnx_model = simplify_onnx_model(onnx_model)
        except Exception as e:
            print(f"Simplifier failure: {e}")
    
    onnx.save(onnx_model, save_path)
    print(f"ONNX export success, saved as {save_path}")


if __name__ == '__main__':
    main(parse_args())