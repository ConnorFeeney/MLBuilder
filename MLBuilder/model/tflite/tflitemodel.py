from typing import Union
import numpy as np
import cv2
from MLBuilder.model.model import MLModel


class TFLiteModel(MLModel):
    def __init__(self, path: str):
        super().__init__(path)

    def build(
        self,
        outdir: str,
        imgsz: int,
        quant: Union[str, None],
        data: str = "coco8.yaml",
        edge: bool = False,
    ) -> str:
        import os
        import shutil
        from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

        if not os.path.isdir(outdir):
            raise ValueError(f"'{outdir}' does not exsist")
        if not os.access(outdir, os.W_OK):
            raise RuntimeError(f"'{outdir} is not writable'")

        working_dir = os.getcwd()
        archive_dir = os.path.join(outdir, "build")
        if not os.path.isdir(archive_dir):
            os.mkdir(archive_dir)
        os.chdir(archive_dir)

        model = YOLO(self.path)
        if not edge:
            model_out = model.export(
                format="tflite",
                imgsz=imgsz,
                half=True if quant == "fp16" else False,
                int8=True if quant == "int8" else False,
                nms=True,
                data=data,
            )
        else:
            model_out = model.export(format="edgetpu", imgsz=imgsz)

        model_dir = os.path.join(archive_dir, model_out)
        os.chdir(working_dir)

        export_dir = os.path.join(outdir, "export")
        if not os.path.isdir(export_dir):
            os.mkdir(export_dir)
        shutil.copy(model_dir, export_dir)

        model_out = os.path.join(export_dir, os.path.basename(model_dir))
        return model_out

    def allocate(self):
        import tensorflow as tf

        self._interpreter = tf.lite.Interpreter(model_path=self.path)
        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()

    def run_inference(self, data: np.ndarray):
        input_h, input_w = self._input_details[0]["shape"][1:3]
        orginal_h, orginal_w = data.shape[:2]

        img_scale = min(input_h / orginal_h, input_w, orginal_w)
        new_h, new_w = int(orginal_h * img_scale), int(orginal_w * img_scale)

        resized_img = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_h = (input_h - new_h) // 2
        pad_w = (input_w - new_w) // 2

        input_img = cv2.copyMakeBorder(
            resized_img,
            top=pad_h,
            bottom=input_h - new_h - pad_h,
            left=pad_w,
            right=input_w - new_w - pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
