from . import _version
from .riftcorner import main as rift_corner_plot
from .convenience import make_header_block

__version__ = _version.get_versions()["version"]
