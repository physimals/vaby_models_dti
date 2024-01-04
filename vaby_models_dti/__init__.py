try:
    from ._version import __version__, __timestamp__
except ImportError:
    __version__ = "Unknown version"
    __timestamp__ = "Unknown timestamp"

from .ballandstick import BallAndStickModel

__all__ = [
    "BallAndStickModel",
    "__version__"
]
