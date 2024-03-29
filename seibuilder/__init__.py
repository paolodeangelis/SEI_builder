"""SEI Builder package with fuctions used in the jupyter."""

from .core.functions import (
    _find_nearest,
    from_d_to_grain,
    get_bulk_atoms,
    get_gcd_pedices,
    get_stable_crystal,
    random_sei_grains,
)

__version__ = "0.1.0"
__copyright__ = "2022, SMaLL Team"
__author__ = "Paolo De Angelis, Roberta Cappabianca and Eliodoro Chiavazzo"

__all__ = [
    "_find_nearest",
    "get_gcd_pedices",
    "from_d_to_grain",
    "get_stable_crystal",
    "random_sei_grains",
]
