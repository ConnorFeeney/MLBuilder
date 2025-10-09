from abc import ABC, abstractmethod
from functools import wraps
from typing import Union


def allow(*allowed_modes):
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
            return func(*args, **kwargs)

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
            return func(*args, **kwargs)

        return wrapper

    return decorator


class MLModel(ABC):
    @staticmethod
    def get_model_type(model_name: str) -> str:
        try:
            model_t = model_name.split(".")[1]
            if not model_t:
                raise ValueError(f"Cant extract model type from '{model_name}'")
            return model_t
        except Exception as e:
            raise ValueError(str(e))

    @property
    @abstractmethod
    def model_type(self) -> str:
        return "pt"

    @abstractmethod
    def __init__(self, path: str, imgsz: int, quant: Union[str, None]):
        self.path: str = path
        self.quant: Union[str, None] = quant
        self.imgsz: Union[int, None] = imgsz

    @allow("pt")
    @abstractmethod
    def build(self, outdir: str, data: str = "coco8.yaml") -> str:
        pass
