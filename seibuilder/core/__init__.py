"""
Core subpackage with SEI builder functions.

It contains functions for getting SEI's salts crystal unit cells, building grains,
and identifying the atoms at the boundary.
"""

from .functions import (
    from_d_to_grain,
    get_gcd_pedices,
    get_stable_crystal,
    random_sei_grains,
)

__all__ = ["from_d_to_grain", "get_stable_crystal", "random_sei_grains", "get_gcd_pedices"]
