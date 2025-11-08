from abc import ABC, abstractmethod
from functools import wraps
from typing import Union
import os


def allow(*allowed_modes: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            model_type = getattr(self, "model_type", "")
            if not model_type:
                raise AttributeError(f"Attribute model_type must be defined")
            if model_type not in allowed_modes:
                raise AttributeError(
                    f"Attribute '{func.__name__}' is not avalible for '{model_type}'"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def disallow(*disallowed_modes):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            model_type = getattr(self, "model_type", "")
            if not model_type:
                raise AttributeError(f"Attribute model_type must be defined")
            if model_type in disallowed_modes:
                raise AttributeError(
                    f"Attribute '{func.__name__}' is not avalible for '{model_type}'"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class MLModel(ABC):
    @staticmethod
    def get_model_type(model_name: str) -> str:
        try:
            model_t = os.path.basename(model_name).split(".")[1]
            if not model_t:
                raise ValueError(f"Cant extract model type from '{model_name}'")
            return model_t
        except Exception as e:
            raise ValueError(str(e))

    @property
    def model_type(self) -> str:
        return self._model_type

    @abstractmethod
    def __init__(self, path: str):
        self.path: str = path
        self._model_type = MLModel.get_model_type(path)

    @allow("pt")
    @abstractmethod
    def build(
        self,
        outdir: str,
        imgsz: int,
        quant: Union[str, None],
        data: str = "coco8.yaml",
    ) -> str:
        pass

    @disallow("pt")
    @abstractmethod
    def allocate(self):
        pass

    @allow("pt")
    def train(
        self, data: str, epoch: int, imgsz: int, outname: str, outdir: str
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
        results = model.train(data=data, epochs=epoch, imgsz=imgsz)

        # Get the best.pt file from the training results
        best_model_path = os.path.join(
            results.save_dir, "weights", "best.pt"  # type: ignore
        )

        os.chdir(working_dir)

        export_dir = os.path.join(outdir, "export")
        if not os.path.isdir(export_dir):
            os.mkdir(export_dir)

        # Copy best.pt to export dir with the specified name
        output = os.path.join(export_dir, outname)
        shutil.copy(best_model_path, output)

        return str(output)
