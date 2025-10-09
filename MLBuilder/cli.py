import argparse
from argparse import Namespace


def build(args: Namespace):
    model_type: str = args.type

    model_name: str = args.model
    imgsz: int = args.imgsz
    model_quant: str = args.quant
    model_edge: bool = args.edge

    outdir = args.out

    if model_type in ("all", "tflite"):
        print("Building TFLite model")
        from MLBuilder.model.tflite.tflitemodel import TFLiteModel

        model = TFLiteModel(model_name, imgsz, model_quant, model_edge)
        outdir = model.build(outdir)
        print(outdir)
    if model_type in ("all", "onnx"):
        print("Building ONNX model")


def run(args: Namespace):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog="MLBuilder", description="Train and run TFLite/ONNX Models"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Build Commands
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument(
        "--model", "-m", help="Base yolo.pt to convert", type=str, default="yolo11n.pt"
    )
    build_parser.add_argument(
        "--type",
        "-t",
        help="Type of model to export (TFLite/ONNX)",
        type=str,
        choices=["all", "tflite", "onnx"],
        default="all",
    )
    build_parser.add_argument(
        "--quant",
        "-q",
        help="Level of optimization",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
    )
    build_parser.add_argument(
        "--imgsz", "-i", help="Model input image size", type=int, default=640
    )
    build_parser.add_argument(
        "--edge",
        "-e",
        action="store_true",
        help="Compile for edge TPU when using TFLite",
    )
    build_parser.add_argument(
        "--out", "-o", help="Relative model output directory", type=str, default="./"
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("name", help="Relative location of model", type=str)

    args = parser.parse_args()

    if args.command == "build":
        build(args)
    if args.command == "run":
        run(args)


if __name__ == "__main__":
    main()
