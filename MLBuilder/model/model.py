def mode(*allowed_modes):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            current_mode = getattr(self, "mode", None)
            if current_mode not in allowed_modes:
                raise PermissionError(
                    f"'{func.__name__}' allowed in {allowed_modes}, "
                    f"but current mode is '{current_mode}'."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
