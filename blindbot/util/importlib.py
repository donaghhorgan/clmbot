import importlib
import types
from typing import Any, Type, TypeVar, Union

T = TypeVar("T")


def get_module_attr(module: Union[str, types.ModuleType], attr: str) -> Any:
    if isinstance(module, str):
        module = importlib.import_module(module)

    obj = getattr(module, attr)

    if obj is None:
        raise ValueError(f"Could not find attribute '{attr}' in module '{module}'")

    return obj


def get_module_type(
    module: Union[str, types.ModuleType], attr: str, type_: Type[T]
) -> Type[T]:
    obj = get_module_attr(module, attr)

    if not issubclass(obj, type_):
        raise ValueError(
            f"Expected '{attr}' to have type '{type_}' " f"but got '{obj}'"
        )

    return obj
