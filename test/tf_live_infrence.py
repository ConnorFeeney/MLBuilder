import sys
from pathlib import Path

import cv2
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        prog="tflive", description="TF Live Infrence Model Test"
    )

    parser.add_argument("model", help="TFlite model to run", type=str)
    parser.add_argument(
        "--video",
        "-v",
        help="What video source to use (defaults to default camera)",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--nms",
        "-n",
        help="Apply NMS post processing (add if not compiled in model)",
        action="store_false",
    )
    parser.add_argument(
        "--tpu",
        "-t",
        help="Attempt to delegate to Coral TPU (Will fall back to cpu if not edge compiled)",
        action="store_false",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        help="Minimum confidence threshold for detections",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()
    try:
        video_source = int(args.video)
    except:
        video_source = args.video
    model_path = args.model
    use_nms = args.nms
    use_tpu = args.tpu
    tolerance = args.confidence

    cap = cv2.VideoCapture(video_source)

    from MLBuilder.model.tflite.tflitemodel import TFLiteModel

    m = TFLiteModel(model_path)
    m.allocate(tpu=use_tpu)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of frame stream.")
            break

        out = m.run_inference(frame, nms=use_nms, tol=tolerance)

        for detection in out:
            bbox = detection["bbox"]
            x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
            x_max, y_max = int(bbox[1][0]), int(bbox[1][1])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
