from typing import Union
from MLBuilder.model.model import MLModel


class TFLiteModel(MLModel):
    @property
    def model_type(self) -> str:
        return self._model_type

    def __init__(
        self, path: str, imgsz: int, quant: Union[str, None], edge: bool = False
    ):
        super().__init__(path, imgsz, quant)
        self._model_type = MLModel.get_model_type(path)
        self.edge = edge

    def build(self, outdir: str, data: str = "coco8.yaml") -> str:
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
        if not self.edge:
            model_out = model.export(
                format="tflite",
                imgsz=self.imgsz,
                half=True if self.quant == "fp16" else False,
                int8=True if self.quant == "int8" else False,
                nms=True,
                data=data,
            )
        else:
            model_out = model.export(format="edgetpu", imgsz=self.imgsz)

        model_dir = os.path.join(archive_dir, model_out)
        os.chdir(working_dir)

        export_dir = os.path.join(outdir, "export")
        if not os.path.isdir(export_dir):
            os.mkdir(export_dir)
        shutil.copy(model_dir, export_dir)

        model_out = os.path.join(export_dir, os.path.basename(model_dir))
        return model_out
