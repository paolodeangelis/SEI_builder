"""SEI Builder package with fuctions used in the jupyter."""

from .functions import (  # get_bulk_atoms, # TODO fix get_bulk_atoms
    _find_nearest,
    _get_gcd_pedices,
    from_d_to_grain,
    get_stable_crystal,
    random_sei_grains,
)

__version__ = "0.1.dev"
