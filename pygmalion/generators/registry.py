"""Generator registry that maps column types to generator classes.

The registry enables the engine to find the correct generator
for each column type without using if/elif chains. New generators
register themselves at import time.
"""

from pygmalion.generators.base import BaseGenerator

_REGISTRY: dict[str, type[BaseGenerator]] = {}


def register(type_name: str, generator_class: type[BaseGenerator]) -> None:
    """Register a generator class for a column type.

    Args:
        type_name: The column type string (e.g., "normal", "uniform").
        generator_class: A class that inherits from BaseGenerator.

    Raises:
        ValueError: If type_name is already registered.
        TypeError: If generator_class is not a subclass of BaseGenerator.
    """
    if type_name in _REGISTRY:
        raise ValueError(f"El tipo '{type_name}' ya está registrado")
    if not issubclass(generator_class, BaseGenerator):
        raise TypeError(f"{generator_class} no es subclase de BaseGenerator")
    _REGISTRY[type_name] = generator_class


def get_generator(type_name: str) -> type[BaseGenerator]:
    """Retrieve the generator class for a column type.

    Args:
        type_name: The column type string to look up.

    Returns:
        The generator class registered for that type.

    Raises:
        KeyError: If no generator is registered for type_name.
    """
    if type_name not in _REGISTRY:
        raise KeyError(f"No hay generador registrado para el tipo '{type_name}'")
    return _REGISTRY[type_name]


def list_registered() -> list[str]:
    """Return a list of all registered column type names.

    Returns:
        List of type name strings currently in the registry.
    """
    return list(_REGISTRY.keys())


def clear_registry() -> None:
    """Remove all entries from the registry.

    Intended for testing only. After calling this, generators
    must be re-registered by reloading their modules.
    """
    _REGISTRY.clear()