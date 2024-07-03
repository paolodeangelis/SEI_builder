"""
This is the SEI builder functions module.

It contains functions for getting SEI's salts crystal unit cells, building grains,
and identifying the atoms at the boundary.
"""

import copy
import re
import signal
from collections import deque
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from ase.atoms import Atoms
from mp_api.client import MPRester
from numpy.random import PCG64, Generator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial import ConvexHull

from ..mpinterfaces import MP_API
from ..mpinterfaces.nanoparticle import Nanoparticle
from ..utils import message

TIME_FORMAT = "%H:%M:%S %Z"
# Material Project query
try:
    matprj = MPRester(MP_API)  # Material Project API KEY
except ValueError:
    pass

matget2ase = AseAtomsAdaptor()


def get_materials_ids(formula: str) -> List[str]:
    """Given a chemical formula retrive the MP ids.

    Args:
        formula (str): Chemical formulas

    Returns:
        List[str]: list of MP ids with that specific chemical formula
    """
    docs = matprj.materials.summary.search(formula=formula, fields=["material_id"])
    mp_ids = [str(m.material_id) for m in docs]
    return mp_ids


def get_structure_by_material_id(material_ids: List[str]) -> List[Structure]:
    """Given a list of the MP ids it retrive the associated Stucture.

    Args:
        material_ids (List[str]): List of MP ids

    Returns:
        List[Structure]: List containg the structures
    """
    docs = matprj.materials.summary.search(material_ids=material_ids, fields=["structure"])
    structures = [m.structure for m in docs]
    return structures


def get_stable_crystal(chem_formula: str) -> Tuple[Atoms, Structure]:
    """
    Get the stable crystal unit-cell from its chemical formula.

    Download from `Materials Project` the cristal file for a given chemical formula.
    If in the database are present metastables configuration, the one with the
    "minimum formation energy per atom" is chosen.

    Args:
        chem_formula (str): Cristal chemical formula (e.g. LiF)

    Returns:
        (tuple): tuple containing:
            -  Strcuture:
               crystal unit as pymatgen object
            -  Atoms:
               crystal unit as ASE object
    """
    mat = get_materials_ids(chem_formula)
    mat = np.array(mat)
    E = []
    for id_ in mat:
        # print(id_)
        E.append(
            matprj.materials.summary.search(material_ids=[id_], fields=["formation_energy_per_atom"])[
                0
            ].formation_energy_per_atom
        )
    E = np.array(E)
    idx = np.where(E == E.min())[0]  # type: ignore
    # Gets stable structure
    structure = get_structure_by_material_id(mat[idx])[0]
    # convert in ase
    structure_ase = matget2ase.get_atoms(structure)
    return structure, structure_ase


# Compute Volume from point cluster


def _tetrahedron_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    return np.abs(np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d))) / 6


def _convex_hull_volume(pts: np.ndarray) -> float:
    ch = ConvexHull(pts)
    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex), ch.simplices))
    tets = ch.points[simplices]
    return np.sum(_tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))


# Grain (Nanoparticle) costructor


def _raise_timeout(signum, frame):
    raise TimeoutError


@contextmanager
def _timeout_contex(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, _raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def from_d_to_grain(
    d: np.ndarray,
    generator: Structure,
    planes: list,
    surface_energies: list,
    maxiter: int = 20,
    tol: float = 0.05,
    timeout: int = 60,
    verbose: int = 0,
) -> Tuple[int, float, float, float, Atoms]:
    """Get iterative the closest grain given a characteristic size.

    Args:
        d (np.ndarray): target characteristic size
        generator (pymatgen.Structure): pymatgen.Structure represents the crystal unit cells.
        planes (list): List of Miller indices corresponding to the cutting planes to use.
        surface_energies (list): surface energy corresponds to each `planes`
        maxiter (int, optional): maximum iteration number for the secant algorithm. Defaults to 20.
        tol (float, optional): convergence tolerance. Defaults to 0.05.
        timeout (int, optional): max waiting time in seconds. Defaults to 60.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages. Defaults to 2.

    Returns:
        Tuple[int, float, float, float, Atoms]: tuple containing:
            Tuple[int, float, float, float, Atoms]: tuple containg:
                -  int: number of atoms for the optimal grain
                -  float: grain volume
                -  float: grain caracteristic lenght
                -  float: maximum radium
                -  ASE.Atoms: computed grain
    """
    rmax_i = [d / 2, d / 2]
    d_i = [0, 0]
    i = 0

    with _timeout_contex(timeout):
        while i < maxiter:
            # generate cluster
            grain = Nanoparticle(generator, rmax=rmax_i[1], hkl_family=planes, surface_energies=surface_energies)
            grain.create()
            grain = matget2ase.get_atoms(grain)
            vol = _convex_hull_volume(grain.positions)
            d_i[1] = np.power(vol * 6 / np.pi, 1 / 3)
            N = grain.positions.shape[0]
            if verbose >= 2:
                message(
                    "Iteration {:d}: Natoms={:d}, rmax={:.2f} A, d={:.2f} A, V={:.2f} AA^3".format(
                        i, N, rmax_i[1], d_i[1], vol
                    ),
                    msg_type="i",
                    add_date=True,
                )
            if i < 1:
                d_i[0] = d_i[1]
                rmax_i[0] = rmax_i[1]
                rmax_i[1] = int(rmax_i[1] + np.sign(d - d_i[0]) * rmax_i[0] * 0.50)
            else:
                if np.abs(d_i[1] - d) <= tol * d:
                    if verbose >= 2:
                        message(
                            r"Convergence after {:d} iterations: Natoms={:d}, rmax={:.2f} A, d={:.2f} A (error {:.2f}"
                            r" %), V={:.2f} AA^3".format(i, N, rmax_i[1], d_i[1], (d - d_i[1]) / d * 100, vol),
                            msg_type="i",
                            add_date=True,
                        )
                    break
                else:
                    # Secant method
                    if np.abs(d_i[1] - d_i[0]) <= d_i[0] * 0.001:
                        if verbose >= 2:
                            message(
                                r"Stall after {:d} iterations: Natoms={:d}, rmax={:.2f} A, d={:.2f} A (error {:.2f} %),"
                                r" V={:.2f} AA^3".format(i, N, rmax_i[1], d_i[1], (d - d_i[1]) / d * 100, vol),
                                msg_type="w",
                                add_date=True,
                            )
                        break
                rmax_next = rmax_i[1] - (rmax_i[1] - rmax_i[0]) / (d_i[1] - d_i[0]) * (d_i[1] - d)
                d_i[0] = d_i[1]
                rmax_i[0] = rmax_i[1]
                rmax_i[1] = rmax_next
            i += 1
    return N, vol, d_i[1], rmax_i[1], grain


def get_gcd_pedices(formula: str) -> int:
    """Get the greatest common divisor (GCD) from number of each atom type in a empirical chemical formula.

    Args:
        formula (str): empirical chemical formula (e.g. Glucose: C6H12O6 ).

    Returns:
        int: the greatest common divisor from number of each atom type.
    """
    pedices = re.findall("[0-9]+", formula)
    pedices = [int(i) for i in pedices]
    return np.gcd.reduce(pedices)


def _print_imposible_grain_warning(grain_id, specie_formula, d_guess, V_guess, surfaces, esurf):
    message(" " * 150, end="\r", msg_type="i", add_date=True)
    message(
        "Was impossible build a grain particle ({:d}) of specie {:s} with target lenght {:.2f} AA and volume"
        " {:.2f} AA^3 and surface ".format(grain_id, specie_formula, d_guess, V_guess) + str(surfaces) + str(esurf),
        end="\n",
        msg_type="w",
        add_date=True,
    )


def random_sei_grains(
    Natoms: int,
    species_unitcell: List[Structure],
    species_fractions: list,
    random_sampler: List[Callable],
    species_fraction_tol: float = 0.005,
    Ngrains_max: int = None,
    report: Optional[str] = None,
    cutting_planes: Optional[list] = None,
    n_planes: int = 2,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Atoms]]:  # type: ignore
    """Get the different grains knowing the size distribution, cutting planes, and molar fraction.

    The function random generates a grain knowing the crystal unit cell and the grain size distribution.
    The sampling is constrained to get the final number of atoms close to ``Natoms`` and to respect the
    molar fraction of each component (``species_fractions``).
    Args:
        Natomas (int): Max number of atoms to sample.
        species_unitcell (List[Structure]): list containing pymatgen.Structure objects that represent
        the unit cells of each SEI crystal component.
        species_fractions (list): The molar fraction for each crystal.
        random_sampler (List[Callable]): list containing the size distribution functions, i.e., a callable
            object that returns the characteristic grain size (diameter).
        species_fraction_tol (float, optional): tolerance for final molar fraction. Defaults to 0.005.
        Ngrains_max (int, optional): Max number of atoms for each grain, if None will be set as 1/10 of ``Natoms``.
            Defaults to None.
        report (str | None, optional): report ``.csv`` file name. If None, the report will be saved in the file
            ``report_grains_sei.csv``. Defaults to None.
        cutting_planes (list | None, optional): list of Miller indices corresponding to the cutting planes to use.
        Defaults to None.
        n_planes (int, optional): number of planes to randomly choose from ``cutting_planes``. Defaults to 2.
        seed (int, optional): random state seed for the random number generator. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, NumPy.ndarray, NumPy.ndarray, NumPy.ndarray, List[Atoms]]: tuple containing:
            -  NumPy.ndarray:
               an array containing the index of the cristal sampled.
            -  NumPy.ndarray:
               an array containing grain size sampled.
            -  NumPy.ndarray:
               an array containing grain volume computed as the "convex hull" volume from the atoms' positions.
            -  NumPy.ndarray:
               array with the final molar fraction for each cristal.
            -  NumPy.ndarray[Atoms]:
               a list containing ASE.Atoms are the random grains generated.
    """
    if Ngrains_max is None:
        Ngrains_max = Natoms // 10

    if species_fractions.sum() > 1.0:  # type: ignore
        species_fractions = species_fractions / species_fractions.sum()  # type: ignore

    if report is None:
        report = "report_grains_sei.csv"

    if report:
        report_file = open(report, "w")
        report_file.write("n;mol;atoms;d;vol;surfaces;surf_energies\n")

    if cutting_planes is None:
        cutting_planes = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]

    # Random Generator
    # seed = 1993
    pcg64 = PCG64(seed=seed)
    rg = Generator(pcg64)
    rg.standard_normal()

    # Inizialization
    out_grains = deque()  # type: ignore # faster access memory if you append a lot
    out_vol = np.zeros(Ngrains_max, dtype=float)
    out_d = np.zeros(Ngrains_max, dtype=float)
    out_species = np.zeros(Ngrains_max, dtype=int)
    molecules_in_unit_cell = np.array([get_gcd_pedices(system.formula) for system in species_unitcell])
    atoms_in_unit_cell = np.array([len(system.sites) for system in species_unitcell])
    species_fractions_atoms = species_fractions / molecules_in_unit_cell * atoms_in_unit_cell
    species_fractions_atoms /= species_fractions_atoms.sum()
    Natoms_per_species = np.ceil(species_fractions_atoms * Natoms).astype(int)
    Natoms_per_species_out = np.zeros(len(species_fractions), dtype=int)
    exceed_per_species = np.array([False] * len(species_fractions))

    i = 0
    while i < Ngrains_max:
        i_specie = rg.choice(np.arange(len(species_fractions)), size=1, p=species_fractions)[0]
        if exceed_per_species[i_specie]:
            continue

        d_guess = random_sampler[i_specie]()
        V_guess = np.power(d_guess, 3) * np.pi / 6
        # unit_cell_vol = np.linalg.det(LiF.lattice_basis)
        specie = species_unitcell[i_specie]
        specie_formula = specie.formula
        unit_cell_vol = specie.lattice.volume
        surfaces = rg.choice(cutting_planes, size=n_planes, replace=False)
        surfaces = [tuple(s) for s in surfaces]
        esurf = np.abs(rg.normal(loc=1, scale=0.2, size=n_planes))
        esurf /= esurf.max()
        # check
        if V_guess < 1.5 * unit_cell_vol:
            message(" " * 150, end="\r", msg_type="i", add_date=True)
            message(
                f"d_guess={d_guess:.2f} AA, vol_guess={V_guess:.2f}  AA^3 too small ({specie_formula:s})",
                end="\r",
                msg_type="i",
                add_date=True,
            )
            continue
        filling_percentage = Natoms_per_species_out / Natoms_per_species * 100
        filling_string = [f"{spec.formula:s}: {filling_percentage[i]:.2f}%" for i, spec in enumerate(species_unitcell)]
        message(" " * 150, end="\r", msg_type="i", add_date=True)
        message(
            "GRAIN {:d}: d_guess={:.2f} AA, vol_guess={:.2f}  AA^3 ({:s}); filling: ".format(
                i + 1, d_guess, V_guess, specie_formula
            )
            + "; ".join(filling_string)
            + " " * 10,
            end="\r",
            msg_type="i",
            add_date=True,
        )
        try:
            N_atoms, Vol_fin, D_fin, rmax_i, grain = from_d_to_grain(
                d_guess, specie, surfaces, esurf, maxiter=20, verbose=0, tol=0.01, timeout=50
            )
        except Exception:  # TODO: get right Exception (too general)
            _print_imposible_grain_warning(i + 1, specie_formula, d_guess, V_guess, surfaces, esurf)
            continue
        message(" " * 150, end="\r", msg_type="i", add_date=True)
        message(f"NP {i + 1:d}: vol={Vol_fin:.2f}, d={D_fin:.2f} ", end="\r", msg_type="i", add_date=True)

        # check species fraction
        if Natoms_per_species_out[i_specie] + N_atoms > Natoms_per_species[i_specie] * (1 + species_fraction_tol):
            exceed_per_species[i_specie] = Natoms_per_species_out[i_specie] >= Natoms_per_species[i_specie] * (
                1 - species_fraction_tol
            )
            message(" " * 150, end="\r", msg_type="w", add_date=True)
            message(
                "Grain particle {:d} rejected since exceed the total numebr of atoms allow for specie {:s}".format(
                    i + 1, specie_formula
                ),
                end="\n",
                msg_type="w",
                add_date=True,
            )
        else:
            if N_atoms > 0 and isinstance(grain, Atoms):
                Natoms_per_species_out[i_specie] += N_atoms
                out_vol[i] = Vol_fin
                out_d[i] = D_fin
                out_species[i] = i_specie
                grain.info["name"] = "grain_{}_{}".format(i, specie_formula.replace(" ", ""))
                grain.info["surface"] = surfaces
                grain.info["surface_energy"] = esurf
                out_grains.append(grain)
                if report:
                    report_file.write(
                        "{:d};{:s};{:d};{:10.6f};{:10.6f};{};{}\n".format(
                            i, specie_formula, len(grain), D_fin, Vol_fin, surfaces, esurf
                        )
                    )
                i += 1
            else:
                _print_imposible_grain_warning(i + 1, specie_formula, d_guess, V_guess, surfaces, esurf)
        # ending criteria
        if np.all(exceed_per_species) or Natoms_per_species_out.sum() > Natoms * (1 + species_fraction_tol):
            break
    message(" " * 150, end="\r", msg_type="i", add_date=True)
    message("END", end="\n", msg_type="i", add_date=True)
    out_grains = list(out_grains)[:i]  # type: ignore
    out_grains = np.array(out_grains, dtype=object)
    # # additional schuffle
    # rg.shuffle(out_grains)
    out_vol = out_vol[:i]
    out_d = out_d[:i]
    out_species = out_species[:i]
    out_species_fraction = Natoms_per_species_out / atoms_in_unit_cell * molecules_in_unit_cell
    out_species_fraction /= out_species_fraction.sum()
    if report:
        report_file.close()
    return out_species, out_d, out_vol, out_species_fraction, out_grains  # type: ignore


# coordination number


def _minmax_rescale(array):
    return (array - array.min()) / (array.max() - array.min())


def _compute_neighborlist(system: Atoms, cutoff: float = 7.5) -> deque:
    nl_list = deque()  # type: ignore # faster access memory if you append a lot
    tot_indices = np.arange(len(system))
    for ai in range(len(system)):
        distances = system.get_distances(ai, tot_indices, mic=False)
        indices = np.where(distances <= cutoff)[0]
        nl_list.append(indices)
    return nl_list


def _compute_score_coordination(neighborlist: Union[list, deque]) -> np.ndarray:
    score = np.array([len(nl_) for nl_ in neighborlist])
    score = _minmax_rescale(score)
    return score


# def _compute_score_steinhardt(system):
#     system.calculate_q([6])
#     score = np.array(system.get_qvals(6))
#     score = 1 - _minmax_rescale(score)
#     return score


def get_bulk_atoms(atoms: Atoms, threshold: float = 0.6, cutoff: float = 7.5) -> Tuple[np.ndarray, np.ndarray]:
    """Identify the atoms in bulk or at the surface.

    Args:
        atoms (Atoms): system to analyze as ASE.Atoms object.
        threshold (float, optional): relative coordination threshold.
            If the scaled coordination number of atoms is above the threshold is identified
            as a bulk atom. Defaults to 0.6.
        cutoff (float, optional): cutoff radius for each atom. Defaults to 7.5.

    Returns:
        Tuple[Numpy.ndarray, Numpy.ndarray]: tuple containing:
            -  Numpy.ndarray:
               boolean array with True the bulk atoms and False the surface atoms.
            -  Numpy.ndarray :
               an array containing the scaled coordination number.
    """
    temp_ase = copy.deepcopy(atoms)
    # check cell
    if np.all(temp_ase.get_cell() == 0):
        temp_ase.set_cell([1e99, 1e99, 1e99])
    # Get neighbors
    neighborlist = _compute_neighborlist(temp_ase, cutoff=cutoff)
    score = _compute_score_coordination(neighborlist)
    bulk_atoms = score > threshold
    return bulk_atoms, score


def _find_nearest(arr, value):
    diff = np.absolute(arr - value)
    index = int(diff.argmin())
    return index
