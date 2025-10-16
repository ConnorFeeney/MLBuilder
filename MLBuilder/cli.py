import argparse
from argparse import Namespace


def build_tflite(args: Namespace):
    """Build TFLite model with specified options."""
    from MLBuilder.model.tflite.tflitemodel import TFLiteModel

    model_name: str = args.model
    imgsz: int = args.imgsz
    model_quant: str = args.quant
    model_edge: bool = args.edge
    outdir: str = args.out
    data: str = args.data

    print("Building TFLite model")
    model = TFLiteModel(model_name)
    outdir = model.build(outdir, imgsz, model_quant, data, model_edge)
    print(f"Model saved to: {outdir}")


def build_onnx(args: Namespace):
    """Build ONNX model with specified options."""
    model_name: str = args.model
    imgsz: int = args.imgsz
    opset: int = args.opset
    simplify: bool = args.simplify
    dynamic: bool = args.dynamic
    outdir: str = args.out

    print("Building ONNX model")
    # TODO: Implement ONNX model building
    print(f"Model: {model_name}, Image size: {imgsz}, Opset: {opset}")
    print(f"Simplify: {simplify}, Dynamic: {dynamic}, Output: {outdir}")


def run(args: Namespace):
    """Run inference with a model."""
    pass


def main():
    parser = argparse.ArgumentParser(
        prog="MLBuilder", description="Build and run TFLite/ONNX Models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command with sub-subparsers for TFLite and ONNX
    build_parser = subparsers.add_parser("build", help="Build models")
    build_subparsers = build_parser.add_subparsers(
        dest="model_type", help="Model type to build"
    )

    # TFLite build parser
    tflite_parser = build_subparsers.add_parser("tflite", help="Build TFLite model")
    tflite_parser.add_argument(
        "--model", "-m", help="Base model to convert", type=str, default="yolo11n.pt"
    )
    tflite_parser.add_argument(
        "--imgsz", "-i", help="Model input image size", type=int, default=640
    )
    tflite_parser.add_argument(
        "--quant",
        "-q",
        help="Quantization level",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
    )
    tflite_parser.add_argument(
        "--edge",
        "-e",
        action="store_true",
        help="Compile for Edge TPU",
    )
    tflite_parser.add_argument(
        "--data",
        "-d",
        help="Dataset YAML for calibration (required for int8)",
        type=str,
        default="coco8.yaml",
    )
    tflite_parser.add_argument(
        "--out", "-o", help="Model output directory", type=str, default="./"
    )

    # ONNX build parser
    onnx_parser = build_subparsers.add_parser("onnx", help="Build ONNX model")
    onnx_parser.add_argument(
        "--model", "-m", help="Base model to convert", type=str, default="yolo11n.pt"
    )
    onnx_parser.add_argument(
        "--imgsz", "-i", help="Model input image size", type=int, default=640
    )
    onnx_parser.add_argument(
        "--opset",
        help="ONNX opset version",
        type=int,
        default=12,
    )
    onnx_parser.add_argument(
        "--simplify",
        "-s",
        action="store_true",
        help="Simplify ONNX model",
    )
    onnx_parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes",
    )
    onnx_parser.add_argument(
        "--out", "-o", help="Model output directory", type=str, default="./"
    )

    # Run parser
    run_parser = subparsers.add_parser("run", help="Run inference with a model")
    run_parser.add_argument("name", help="Relative location of model", type=str)

    args = parser.parse_args()

    if args.command == "build":
        if args.model_type == "tflite":
            build_tflite(args)
        elif args.model_type == "onnx":
            build_onnx(args)
        else:
            build_parser.print_help()
    elif args.command == "run":
        run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
