"""
This is the SEI builder functions module.

It contains functions for getting SEI's salts crystal unit cells, building grains,
and identifying the atoms at the boundary.
"""
import re
import signal
from collections import deque
from contextlib import contextmanager
from typing import Callable, List, Tuple

import numpy as np

# import pyscal.core as pc # TODO remobe pyscal
from ase.atoms import Atoms
from numpy.random import PCG64, Generator
from pymatgen import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial import ConvexHull

from ..mpinterfaces import MP_API
from ..mpinterfaces.nanoparticle import Nanoparticle
from ..utils import message

TIME_FORMAT = "%H:%M:%S %Z"
# Material Project query
matprj = MPRester(MP_API)  # Material Project API KEY
matget2ase = AseAtomsAdaptor()


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
    mat = matprj.get_materials_ids(chem_formula)
    mat = np.array(mat)
    propertie = "formation_energy_per_atom"
    E = []
    for id_ in mat:
        E.append(matprj.query(criteria={"task_id": id_}, properties=[propertie])[0][propertie])
    E = np.array(E)
    idx = np.where(E == E.min())[0]
    # Gets stable structure
    structure = matprj.get_structure_by_material_id(mat[idx][0])
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
    generator: np.random.Generator,
    surfaces: list,
    surface_energies: list,
    maxiter: int = 20,
    verbose: bool = False,
    tol: float = 0.05,
    timeout: int = 60,
) -> Tuple[int, float, float, float, Atoms]:  # TODO docstring
    """
    _summary_ .

    _description_.

    Args:
        d (np.ndarray): _description_
        generator (np.random.Generator): _description_
        surfaces (list): _description_
        surface_energies (list): _description_
        maxiter (int, optional): _description_. Defaults to 20.
        verbose (bool, optional): _description_. Defaults to False.
        tol (float, optional): _description_. Defaults to 0.05.
        timeout (int, optional): _description_. Defaults to 60.

    Returns:
        (tuple): tuple containing:
            Tuple[int, float, float, float, Atoms]: _description_
    """
    rmax_i = [d / 2, d / 2]
    d_i = [0, 0]
    i = 0

    with _timeout_contex(timeout):
        while i < maxiter:
            # generate cluster
            grain = Nanoparticle(generator, rmax=rmax_i[1], hkl_family=surfaces, surface_energies=surface_energies)
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
        ing: the greatest common divisor from number of each atom type.
    """
    pedices = re.findall("[0-9]+", formula)
    pedices = [int(i) for i in pedices]
    return np.gcd.reduce(pedices)


def random_sei_grains(
    Natoms: int,
    species: List[Structure],
    species_fractions: list,
    random_sampler: List[Callable],
    species_fraction_tol: float = 0.005,
    Ngrains_max: int = None,
    report: str or None = None,
    cutting_planes: list or None = None,
    n_planes: int = 2,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Atoms]]:  # TODO `random_sei_grains` docstring
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
        report (str or None, optional): report ``.csv`` file name. If None, the report will be saved in the file
            ``report_grains_sei.csv``. Defaults to None.
        cutting_planes (list or None, optional): _description_. Defaults to None.
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
            -  List[Atoms]:
               a list containing ASE.Atoms are the random grains generated.
    """
    if Ngrains_max is None:
        Ngrains_max = Natoms // 10

    if species_fractions.sum() > 1.0:
        species_fractions = species_fractions / species_fractions.sum()

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
    out_grains = deque()  # faster access memory if you append a lot
    out_vol = np.zeros(Ngrains_max, dtype=float)
    out_d = np.zeros(Ngrains_max, dtype=float)
    out_species = np.zeros(Ngrains_max, dtype=int)
    molecules_in_unit_cell = np.array([get_gcd_pedices(i.formula) for i in species])
    atoms_in_unit_cell = np.array([len(i.sites) for i in species])
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
        specie = species[i_specie]
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
        filling_string = [f"{spec.formula:s}: {filling_percentage[i]:.2f}%" for i, spec in enumerate(species)]
        message(" " * 150, end="\r", msg_type="i", add_date=True)
        message(
            "NP {:d}: d_guess={:.2f} AA, vol_guess={:.2f}  AA^3 ({:s}); filling: ".format(
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
        except Exception:
            message(" " * 150, end="\r", msg_type="i", add_date=True)
            message(
                "Was impossible build a nanoparticle ({:d}) of specie {:s} with target lenght {:.2f} AA and volume"
                " {:.2f} AA^3 and surface ".format(i + 1, specie_formula, d_guess, V_guess)
                + str(surfaces)
                + str(esurf),
                end="\n",
                msg_type="w",
                add_date=True,
            )
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
                "Nanoparticle {:d} rejected since exceed the total numebr of atoms allow for specie {:s}".format(
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
                message(" " * 150, end="\r", msg_type="i", add_date=True)
                message(
                    "Was impossible build a nanoparticle ({:d}) of specie {:s} with target lenght {:.2f} AA and volume"
                    " {:.2f} AA^3 and surface ".format(i + 1, specie_formula, d_guess, V_guess)
                    + str(surfaces)
                    + str(esurf),
                    end="\n",
                    msg_type="w",
                    add_date=True,
                )
        # ending criteria
        if np.all(exceed_per_species) or Natoms_per_species_out.sum() > Natoms * (1 + species_fraction_tol):
            break
    message(" " * 150, end="\r", msg_type="i", add_date=True)
    message("END", end="\n", msg_type="i", add_date=True)
    out_grains = list(out_grains)[:i]
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
    return out_species, out_d, out_vol, out_species_fraction, out_grains


# coordination number


def _minmax_rescale(array):
    return (array - array.min()) / (array.max() - array.min())


def _compute_score_coordination(system):
    atoms = system.atoms
    score = np.array([atom.coordination for atom in atoms])
    score = _minmax_rescale(score)
    return score


def _compute_score_steinhardt(system):
    system.calculate_q([6])
    score = np.array(system.get_qvals(6))
    score = 1 - _minmax_rescale(score)
    return score


# TODO Get rid of pyscal
# def get_bulk_atoms(
#     cluster, strategy="coordination", method="cutoff", threshold=0.6, cutoff=5.0, **kwarg
# ):
#     """_summary_ .

#     _description_.

#     Args:
#         cluster (_type_): _description_
#         strategy (str, optional): _description_. Defaults to "coordination".
#         method (str, optional): _description_. Defaults to "cutoff".
#         threshold (float, optional): _description_. Defaults to 0.6.
#         cutoff (float, optional): _description_. Defaults to 5.0.

#     Raises:
#         ValueError: _description_

#     Returns:
#         _type_: _description_
#     """
#     temp_ase = copy.deepcopy(cluster)
#     temp = pc.System()
#     # Convert to pystacal.core.System
#     if np.all(temp_ase.get_cell() == 0):
#         temp_ase.set_cell([1e99, 1e99, 1e99])
#     temp.read_inputfile(temp_ase, format="ase")
#     # Get neighbors
#     temp.find_neighbors(method=method, cutoff=cutoff, **kwarg)
#     # Get score
#     if strategy.lower() == "coordination":
#         score = _compute_score_coordination(temp)
#     elif strategy.lower() == "steinhardt":
#         score = _compute_score_steinhardt(temp)
#     else:
#         raise ValueError(f"{strategy} is not a valid strategy.")
#     bulk_atoms = score > threshold
#     return bulk_atoms, score


def _find_nearest(arr, value):
    diff = np.absolute(arr - value)
    index = int(diff.argmin())
    return index
