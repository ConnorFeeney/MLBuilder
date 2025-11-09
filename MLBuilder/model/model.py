from abc import ABC, abstractmethod
from functools import wraps
from typing import Union
import os
import platform


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


def system(required_system: str):
    required_system = required_system.lower()

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            current_system = platform.system().lower()  # "linux", "windows", "darwin"
            if current_system != required_system:
                raise RuntimeError(
                    f"Method '{func.__name__}' requires {required_system.capitalize()}, "
                    f"but current system is {current_system.capitalize()}"
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
        import os

        self.path: str = path
        self._model_type = MLModel.get_model_type(path)

    @allow("pt")
    @system("linux")
    @abstractmethod
    def build(self) -> str:
        pass

    @disallow("pt")
    @system("linux")
    @abstractmethod
    def allocate(self):
        pass

    @system("linux")
    @abstractmethod
    def run_inference(self):
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

        data = os.path.abspath(data) if data != "coco8.yaml" else data
        outdir = os.path.abspath(outdir)

        working_dir = os.getcwd()
        archive_dir = os.path.join(outdir, "build")
        if not os.path.isdir(archive_dir):
            os.mkdir(archive_dir)
        os.chdir(archive_dir)

        model = YOLO(os.path.abspath(self.path))
        results = model.train(
            data=data,
            epochs=epoch,
            imgsz=imgsz,
            project=os.path.join(archive_dir, "out"),
        )

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
