"""
Wrapper for Packmol.

Thank you to Richard Gowers from MDAnalysis,
see https://github.com/MDAnalysis/MDAPackmol
"""

import copy
import os
import time
from typing import List, Optional, Union

import numpy as np
from ase.atoms import Atoms
from ase.io import read, write

from .._constants import JOB_STATUS, JOB_STATUS_INV
from ..run import _follow_job, check_job_status, submit_job, write_script
from ..utils import message

PACKMOL_INP = "packmol.inp"  # name of .inp file given to packmol
PACKMOL_STRUCTURE_FILES = "PACKMOL_MOL_input{}.xyz"
PACKMOL_OUT = "output.xyz"
PACKMOL_LOG = "packmol.log"
TMP_RESNAME = "resname.tmp"
RESIDUE_DEFAULT_NAME = "res_{}"
MOLECULE_DEFAULT_NAME = "mol_{}"


class PackmolStructure:
    """Class to handle ASE ``Atoms`` object in ``packmol``."""

    def __init__(self, atoms: Atoms, number: int, instructions: Union[list, str], resname: str = None):
        """Molecule to add to the Packmol system.

        Args:
            atoms (ASE.Atoms): a single template molecule for Packmol to use.
            number (int): quantity of this molecule to add to new system.
                instructions (list or str): list of instructions to Packmol for this molecule
                eg 'inside box 0. 0. 0. 40. 40. 40.'
                each item in the list should be a single line instruction.
            resname (str, optional): residual name to use. Defaults to None.
        """
        self.atoms = atoms
        self.number = number
        self.instructions = instructions
        self.resname = resname

    def to_packmol_inp(self, index: int) -> str:
        """Create portion of packmol.inp file from this molecule.

        Args:
            index (int): the index of this template molecule within entire system.

        Returns:
            str: text to write to Packmol input for this molecule.
        """
        output = f"structure {PACKMOL_STRUCTURE_FILES.format(index)}\n"
        output += f"  number {self.number}\n"
        for instruction in self.instructions:
            output += "  " + instruction + "\n"
        output += "end structure\n\n"

        return output

    def save_structure(self, index: int):
        """Save this molecule for Packmol to use.

        Args:
            index (int): the index of this template molecule within entire system.
        """
        # we mangle Resnames to keep track of which molecule is which
        # so copy the true names, change, write out, change back to real
        write(PACKMOL_STRUCTURE_FILES.format(index), self.atoms)


def make_packmol_input(
    structures: List[PackmolStructure],
    tolerance: float = 2.0,
    verbose: int = 1,
    maxit: int = 20,
    nloop: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Convert the call into a Packmol usable input file.

    Args:
        structures (list): list of ``PackmolStructure`` objects
        tolerance (float, optional): Minimum distance between molecules. Defaults to 2.0.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages. Defaults to 2.
        maxit (int, optional): This is the maximum number of iterations of the local optimizer (GENCAN) per loop.
            changing it may improve (or worse) the convergence. Defaults to 20.
        nloop (int, optional): Change the maximum number of optimization loops (if this option is used inside
            the structure section, it affects the number of optimization loops only of that specific structure).
            Defaults to None.
        seed (int, optional): Change random number generator seed. Defaults to None.
    """
    with open(PACKMOL_INP, "w") as out:
        out.write("# autogenerated packmol input\n\n")
        out.write(f"tolerance {tolerance}\n\n")
        if seed is not None:
            out.write(f"seed {seed}\n\n")
        if maxit is not None:
            out.write(f"maxit {maxit}\n\n")
        if nloop is not None:
            out.write(f"nloop {nloop}\n\n")
        out.write("filetype xyz\n\n")

        for i, structure in enumerate(structures):
            out.write(structure.to_packmol_inp(i))
            structure.save_structure(i)

        out.write(f"output {PACKMOL_OUT}\n\n")
    # print tmp residual name list
    with open(TMP_RESNAME, "w") as out:
        for i, structure in enumerate(structures):
            resname = structure.resname
            if resname is None:
                resname = RESIDUE_DEFAULT_NAME.format(i + 1)
                if verbose >= 2:
                    message(f"The molecule {i+1} has no a residue name (we will named " + f"'{resname}')", msg_type="w")
            Na = len(structure.atoms)
            Nm = structure.number
            for n in range(Nm):
                for na in range(Na):
                    out.write(f"{resname}_{n}\n")


# def _print_and_write(string, **kwarg):
#     f_ = open(PACKMOL_LOG, "a")
#     f_.write(string.split("[I]")[-1].split("\x1b[3F")[0] + "\n")
#     print(string, **kwarg)
#     f_.close()


# def _print_simple(string, **kwarg):
#     print(string, **kwarg)


def run_packmol(packmol_bin: str = None, n: int = 10, slurm: bool = False, verbose: int = 2):
    """Run and follow that Packmol worked correctly.

    Args:
        packmol_bin (str, optional): path to the ``packmol`` binary if it is not stored in ``PATH``,
            if None will use the command ``packmol``. Defaults to None.
        n (int, optional): number of starting lines to print. Defaults to 10.
        slurm (bool, optional): if True will submit a job with``SLURM`` workload manager,. Defaults to False.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages. Defaults to 2.

    Raises:
        PackmolError: raise if ``packmol`` did not converged.
    """
    # Write script
    if packmol_bin is not None:
        cmd = f"{packmol_bin} < {PACKMOL_INP}"
    else:
        cmd = f"packmol < {PACKMOL_INP}"
    try:
        # write script
        run_script = write_script(cmd=cmd, sim_id="packmol", slurm=slurm, verbose=verbose)
        # Submit
        job_id, job_out_file, job_err_file = submit_job(run_script, sim_id="packmol", slurm=slurm, verbose=verbose)
        # follow
        _follow_job(
            job_id,
            job_out_file,
            job_err_file,
            slurm=slurm,
            n=n,
            add_time=True,
            verbose=verbose,
        )
        status, _ = check_job_status(
            job_id=job_id,
            job_out_file=job_out_file,
            job_err_file=job_err_file,
            slurm=slurm,
            verbose=-1,
        )
        if status == JOB_STATUS["COMPLETED"] or status == JOB_STATUS["COMPLETING"]:
            pass
        else:
            message(f"Job end status = {status} ({JOB_STATUS_INV[status]})", msg_type="e")
            raise PackmolError(f"Job end status = {status} ({JOB_STATUS_INV[status]})")
    except KeyboardInterrupt:
        if verbose >= 0:
            message(" " * 100, msg_type="w", end="\r")
            message("SIGINT signal detected!", msg_type="w")

    os.remove(run_script)
    os.remove(job_out_file)
    os.remove(job_err_file)


def _get_parents(structures, verbose: int = 0):
    parents = []
    for structure in structures:
        parents += [copy.copy(structure.atoms)] * structure.number
    if verbose >= 2:
        message(f"The final `System` has {len(parents)} parents")
    return parents


def _get_resname(verbose: int = 0):
    with open(TMP_RESNAME) as file:
        resnames = file.readlines()
    resnames = [r.replace("\n", "") for r in resnames]
    if verbose >= 2:
        message(f"The final `System` has {len(np.unique(resnames))} residues")
    return resnames


# def _get_bonds(structures, verbose: int = 0):
#     bonds = []
#     N = 0
#     for structure in structures:
#         number = structure.number
#         atoms = structure.atoms
#         n_atoms = len(atoms)
#         if hasattr(atoms, "bonds"):
#             for i in range(number):
#                 for bond_ in atoms.bonds:
#                     bonds.append(
#                         Bond(bond_.i + N, bond_.j + N, bond_.ai, bond_.aj, bond_.d, bo=bond_.bo, type=bond_.type)
#                     )
#                 N += n_atoms
#     if verbose >= 2:
#         message(f"The new `System` has {len(bonds)} bonds')")
#     return bonds


# def get_molecules(structures, atoms, verbose: int = 0):
#     residue = []
#     residue_name = []
#     molecules = []
#     molecules_name = []
#     molecules_id = []
#     indexs = []
#     symbols = atoms.get_chemical_symbols()
#     positions = atoms.get_positions()
#     index = 0
#     for res_i, structure in enumerate(structures):
#         N_mol = structure.number
#         N_atoms = len(structure.atoms)
#         residue.append(structure.atoms.copy())
#         if "name" in residue[-1].info:
#             residue_name.append(residue[-1].info["name"])
#         else:
#             if verbose >= 2:
#                 message(
#                     f"The residue {res_i+1} has no name (we will named " +
#                       f"'{RESIDUE_DEFAULT_NAME.format(res_i+1)}')"
#                 )
#             residue_name.append(RESIDUE_DEFAULT_NAME.format(res_i + 1))
#         for j in range(N_mol):
#             mol_name = residue_name[-1] + "/" + MOLECULE_DEFAULT_NAME.format(j + 1)
#             mol_index = []
#             mol_positions = []
#             mol_symbols = []
#             for k in range(N_atoms):
#                 mol_index.append(index)
#                 mol_positions.append(positions[index])
#                 mol_symbols.append(symbols[index])
#                 index += 1
#             molecules.append(Atoms(symbols=mol_symbols, positions=mol_positions))
#             molecules_name.append(mol_name)
#             molecules_id.append(res_i)
#             indexs.append(mol_index)
#     return indexs, residue, residue_name, molecules, molecules_name, molecules_id


def load_packmol_output(structures: List[PackmolStructure], verbose: int = 0):
    """Load the packmol output.

    Args:
        structures (list): list of ``PackmolStructure`` objects
        (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages. Defaults to 2. Defaults to 0.

    Returns:
        ASE.Atoms: final system with additiona info in the property ``Atoms.info``.
    """
    waiting = True
    while waiting:
        if os.path.exists(PACKMOL_OUT):
            waiting = False
            time.sleep(0.05)
        else:
            time.sleep(0.05)
    # Convert in a system
    atoms = read(PACKMOL_OUT)
    # symbols = atoms.get_chemical_symbols()
    # positions = atoms.get_positions()
    # system = Atoms(symbols=symbols, positions=positions)
    # save parents
    atoms.info["parents"] = _get_parents(structures, verbose=verbose)
    # copy residue
    atoms.info["resname"] = _get_resname(verbose=verbose)
    return atoms


def _clean_tempfiles(structures):
    """Delete files generated by MDAPackmol.

    Args:
        structures (list): list of ``PackmolStructure`` objects.
    """
    structure_files = [PACKMOL_STRUCTURE_FILES.format(i) for i in range(len(structures))]

    for f in [PACKMOL_INP, PACKMOL_OUT, TMP_RESNAME] + structure_files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass


class PackmolError(Exception):
    """Pacmol error class."""

    # if packmol didn't find solution


def packmol(
    structures: List[PackmolStructure],
    packmol_bin: str = None,
    n_lines: int = 10,
    cell: list = None,
    tolerance: float = 2.0,
    slurm: bool = False,
    maxit: int = None,
    nloop: int = None,
    seed: int = None,
    verbose: int = 2,
):
    """Take molecules and settings and create a larger system.

    Args:
        structures (list): list of ``PackmolStructure`` objects.
        packmol_bin (str, optional): path to the ``packmol`` binary if it is not stored in ``PATH``,
            if None will use the command ``packmol``. Defaults to None.
        n_lines (int, optional): number of starting lines to print. Defaults to 10.
        cell (list, optional): cell: 3x3 matrix or length 3 or 6 vector
            Unit cell.  A 3x3 matrix (the three unit cell vectors) or
            just three numbers for an orthorhombic cell. Another option is
            6 numbers, which describes unit cell with lengths of unit cell
            vectors and with angles between them (in degrees), in following
            order: [len(a), len(b), len(c), angle(b,c), angle(a,c),
            angle(a,b)], see https://wiki.fysik.dtu.dk/ase/ase/cell.html?highlight=cell#module-ase.cell.
            Defaults to None.
        tolerance (float, optional): : Minimum distance between molecules. Defaults to 2.0.
        slurm (bool, optional): if True will submit a job with``SLURM`` workload manager,. Defaults to False.
        maxit (int, optional): This is the maximum number of iterations of the local optimizer (GENCAN) per loop.
            changing it may improve (or worse) the convergence. Defaults to 20.
        nloop (int, optional): Change the maximum number of optimization loops (if this option is used inside
            the structure section, it affects the number of optimization loops only of that specific structure).
            Defaults to None.
        seed (int, optional): Change random number generator seed. Defaults to None.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages. Defaults to 2.

    Returns:
        ASE.Atoms: final system with additiona info in the property ``array``.
    """
    try:
        make_packmol_input(structures, tolerance=tolerance, verbose=verbose, maxit=maxit, nloop=nloop, seed=seed)
        run_packmol(packmol_bin=packmol_bin, n=n_lines, slurm=slurm, verbose=verbose)
    except PackmolError:
        # todo: Deal with error
        new = None
    else:
        new = load_packmol_output(structures, verbose=verbose)
        if cell is not None:
            new.set_cell(cell)
        new.set_pbc([True, True, True])
    finally:
        _clean_tempfiles(structures)
        # pass
    return new


# def _job_following(folder_path, logfile_name="ams.log", head=5):
#     i = 0
#     j = 0
#     clock = ["|", "/", "-", "\\"]
#     time.sleep(0.2)
#     while True:
#         try:
#             log_file = glob(os.path.join(folder_path, logfile_name))[-1]
#             log_tail = tail("-f", log_file, _iter=True)
#             print(" " * 180, end="\x1b[3F\x1b[1K\r", flush=True)
#             print(
#                 "\x1b[32m[I]: "
#                 + " [{}]: ".format(datetime.now(timezone("Europe/Rome")).strftime("%H:%M:%S"))
#                 + "Log file found                       \x1b[0m",
#                 flush=True,
#             )
#             break
#         except:
#             print(
#                 "\x1b[33m[I]"
#                 + " [{}]: ".format(datetime.now(timezone("Europe/Rome")).strftime("%H:%M:%S"))
#                 + f"Searching the log file in {folder_path} "
#                 + clock[i % 4],
#                 end="\x1b[3F\x1b[1K\x1b[0m\r",
#                 flush=True,
#             )
#             i += 1
#             time.sleep(0.2)

#     while True:
#         time.sleep(0.001)
#         stout = log_tail.next()
#         if j >= head:
#             print(" " * 180, end="\x1b[3F\x1b[1K\r", flush=True)
#             print(
#                 "\x1b[37m\x1b[40m[I " + log_file + "]: " + stout.split("\n")[0],
#                 end="\x1b[3F\x1b[1K\x1b[0m\r",
#                 flush=True,
#             )
#         else:
#             j += 1
#             print("[I " + log_file + "]: " + stout.split("\n")[0], end="\n")

#         if "ERROR:" in stout:
#             print(" " * 180, end="\x1b[3F\x1b[1K\r", flush=True)
#             print(
#                 "\x1b[37m\x1b[41m[E " + log_file + "]: " + stout.split("\n")[0],
#                 end="\x1b[3F\x1b[1K\x1b[0m\r",
#                 flush=True,
#             )
#             print("\n")
#             break

#         if "TERMINATION" in stout:
#             print(" " * 180, end="\x1b[3F\x1b[1K\r", flush=True)
#             print(
#                 "\x1b[37m\x1b[42m[I " + log_file + "]: " + stout.split("\n")[0],
#                 end="\x1b[3F\x1b[1K\x1b[0m\r",
#                 flush=True,
#             )
#             print("\n")
#             break
