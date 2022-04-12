"""
Wrapper for Packmol.

Thank you to Richard Gowers from MDAnalysis,
see https://github.com/MDAnalysis/MDAPackmol
"""
from .packmol import PackmolStructure, packmol

__all__ = ["PackmolStructure", "packmol"]
