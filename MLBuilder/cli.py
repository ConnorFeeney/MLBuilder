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
    output = model.build(outdir, imgsz, model_quant, data, model_edge)
    print(f"Model saved to: {output}")


def train_tflite(args: Namespace):
    from MLBuilder.model.tflite.tflitemodel import TFLiteModel

    model_name: str = args.model
    data: str = args.data
    epoch: int = args.epoch
    imgsz: int = args.imgsz
    outname: str = args.outname
    outdir: str = args.out

    print("Training TFLite model")
    model = TFLiteModel(model_name)
    output = model.train(
        data=data, epoch=epoch, imgsz=imgsz, outname=outname, outdir=outdir
    )
    print(f"Model saved to: {output}")


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

    train_parser = subparsers.add_parser("train", help="Build models")
    train_subparsers = train_parser.add_subparsers(
        dest="model_type", help="Model type to train"
    )

    tftrain_parser = train_subparsers.add_parser("tflite", help="Train tflite model")
    tftrain_parser.add_argument(
        "--model", "-m", help="Base model to train", type=str, default="yolo11n.pt"
    )
    tftrain_parser.add_argument(
        "--data", "-d", help="Data to train off of", type=str, default="coco8.yaml"
    )
    tftrain_parser.add_argument(
        "--imgsz", "-i", help="Image size", type=int, default=640
    )
    tftrain_parser.add_argument(
        "--epoch", "-e", help="Epochs to train on", type=int, default=10
    )
    tftrain_parser.add_argument(
        "--out", "-o", help="Output directory", type=str, default="./"
    )
    tftrain_parser.add_argument(
        "--outname", "-n", help="Output name", type=str, default="model.pt"
    )

    args = parser.parse_args()

    if args.command == "build":
        if args.model_type == "tflite":
            build_tflite(args)
    elif args.command == "train":
        if args.model_type == "tflite":
            train_tflite(args)


if __name__ == "__main__":
    main()
