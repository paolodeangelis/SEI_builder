"""Job handeler module."""
import os
import time
from datetime import datetime
from typing import Tuple

from IPython.display import clear_output

from .._constants import JOB_STATUS, JOB_STATUS_INV, SLEEP_INTERVAL, TIME_FORMAT
from ..utils import are_we_in_a_notebook, message, pytail_follow
from .bash_handler import check_job_status_bash, submit_job_bash, write_script_bash
from .job_handler_utils import output_job_tail


def write_script(
    cmd: str or list,
    sim_id: str,
    slurm: bool = False,
    nodes: int = 1,
    gpus: int = None,
    ntasks_per_node: int = 1,
    cpus_per_task: int = None,
    ntasks_per_socket: int = None,
    gres: int or str = None,
    script_path: str = None,
    job_settings: str = None,
    verbose: int = 0,
) -> str:
    """Write the JOB script file.

    Args:
        cmd  (str or list): job commands.
        sim_id (str):  simulation id.
        slurm (bool, optional): [NotImplemented] if True will write a SLURM script. Defaults to False.
        nodes (int, optional): [NotImplemented] cores per node (equivalent to SLURM option ``--ntasks-per-node``).
            Defaults to 1.
        gpus (int, optional): [NotImplemented] total number of gpus (equivalent to SLURM option ``--gpus``).
        Defaults to None.
        ntasks_per_node (int, optional): [NotImplemented] request that ntasks be invoked on each node
            (equivalent to SLURM option ``--ntasks-per-node``). Defaults to 1.
        cpus_per_task (int, optional): [NotImplemented] request that ncpus be allocated per process.
            This may be useful if the job is multithreaded and requires more than one CPU per task for optimal
            performance. (equivalent to SLURM option ``--cpus-per-task``). Defaults to None.
        ntasks_per_socket (int, optional): [NotImplemented] request the maximum ntasks be invoked on each socket
            (equivalent to SLURM option ``--ntasks-per-socket``). Defaults to None.
        gres (intorstr, optional): [NotImplemented] specifies a comma-delimited list of generic consumable resources.
            The format of each entry on the list is "name[[:type]:count]". (equivalent to SLURM option ``--gres``)
            Defaults to None.
        script_path (str, optional): _description_. Defaults to None.
        job_settings (str, optional): [NotImplemented] jobs settings file path. Defaults to None.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages
            Defaults to 0.

    Raises:
        NotImplementedError: raise the error if you try to use it with SLURM Workload Manager.

    Returns:
        str: Script file path.
    """
    if slurm:
        raise NotImplementedError("SLURM job handler is not yer implemented, sorry!")
    else:
        script_path = write_script_bash(
            cmd,
            sim_id,
            job_settings=job_settings,
            script_path=script_path,
            verbose=verbose,
        )
    return os.path.abspath(script_path)


def submit_job(
    script_path: str, sim_id: str, slurm: bool = True, afterok: int = None, verbose: int = 0
) -> Tuple[int, str, str]:
    """Submit the job.

    Args:
        script_path (str): the job input script.
        sim_id (str): simulation id.
        slurm (bool, optional): [NotImplemented] if True will write a SLURM script. Defaults to True.
        afterok (int, optional): [NotImplemented] jobid to follow. Defaults to None.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages.
            Defaults to 0.

    Raises:
        NotImplementedError: raise the error if you try to use it with SLURM Workload Manager.

    Returns:
        Tuple[int, str, str]: tuple containing:
            -  int: the job ID.
            -  str: the path where the job `STDOUT` is stored.
            -  str: the path where the job `STDERR` is stored.
    """
    if slurm:
        raise NotImplementedError("SLURM job handler is not yer implemented, sorry!")
    else:
        job_id, job_out_file, job_err_file = submit_job_bash(script_path, sim_id, afterok=afterok, verbose=verbose)
    return job_id, os.path.abspath(job_out_file), os.path.abspath(job_err_file)


def check_job_status(
    job_id: int,
    job_out_file: str,
    job_err_file: str,
    slurm: bool = True,
    verbose: int = 0,
) -> Tuple[str, datetime]:
    """Check job status.

    Args:
        job_id (int): the command PID (equivalent to job ID).
        job_out_file (str): the path where the job `STDOUT` is stored.
        job_err_file (str): the path where the job `STDERR` is stored.
        slurm (bool, optional): [NotImplemented] if True will write a SLURM script. Defaults to False.
        verbose (int, optional): loudness controller:
                -  0: print errors
                -  1: print errors and warnings
                -  2: print errors, warnings and info
                -  3: print errors, warnings, info and debugger messages.
                Defaults to 0.

    Raises:
        NotImplementedError: raise the error if you try to use it with SLURM Workload Manager.

    Returns:
        Tuple[str, datetime]: tuple containing:
            -  str: job status.
            -  datetime: the ending time if completed, None otherwise
    """
    if slurm:
        raise NotImplementedError("SLURM job handler is not yer implemented, sorry!")
    else:
        status, end_date = check_job_status_bash(job_id, job_out_file, job_err_file, verbose=verbose)
    return status, end_date


def _wait_job(
    job_id,
    job_out_file: str,
    job_err_file: str,
    slurm: bool = True,
    add_time: bool = True,
    verbose: int = 0,
):
    status, _ = check_job_status(
        job_id=job_id,
        job_out_file=job_out_file,
        job_err_file=job_err_file,
        slurm=slurm,
        verbose=verbose,
    )
    i = 0
    clock = ["|", "/", "-", "\\"]
    if status != JOB_STATUS["RUNNING"]:
        if verbose >= 2:
            message(" " * 100, msg_type="i", end="\r", flush=True)
            message(
                "Job {} is not running (status: {} ({})) {}".format(
                    job_id,
                    status,
                    JOB_STATUS_INV[status],
                    clock[i % 4],
                ),
                msg_type="i",
                end="\r",
                flush=True,
                add_date=add_time,
            )
        i += 1
        time.sleep(0.25)
        return True
    else:
        if add_time and verbose >= 2:
            message(" " * 100, msg_type="i", end="\r", flush=True)
            message(
                f"[{datetime.now().astimezone().strftime(TIME_FORMAT)}] Job {job_id} is RUNNING",
                msg_type="i",
            )
        elif verbose >= 0:
            message(" " * 100, msg_type="i", end="\r", flush=True)
            message(f"Job {job_id} is RUNNING", msg_type="i")
        return False


def _follow_job(
    job_id,
    job_out_file: str,
    job_err_file: str,
    slurm: bool = True,
    n: int = 10,
    add_time: bool = True,
    verbose: int = 0,
):
    notebook = are_we_in_a_notebook()
    status, _ = check_job_status(
        job_id=job_id,
        job_out_file=job_out_file,
        job_err_file=job_err_file,
        slurm=slurm,
        verbose=-1,
    )
    waiting = True
    i = 0
    clock = ["|", "/", "-", "\\"]
    try:
        while waiting:
            waiting = _wait_job(
                job_id,
                job_out_file=job_out_file,
                job_err_file=job_err_file,
                slurm=slurm,
                add_time=add_time,
                verbose=verbose,
            )
    except KeyboardInterrupt:
        if verbose >= 0:
            message(" " * 100, msg_type="w", end="\r")
            message("SIGINT signal detected!", msg_type="w")

    try:
        status, _ = check_job_status(
            job_id=job_id,
            job_out_file=job_out_file,
            job_err_file=job_err_file,
            slurm=slurm,
            verbose=verbose,
        )
        if notebook and 2 <= verbose < 4:
            waiting = True
            while waiting:
                if os.path.exists(job_out_file):
                    waiting = False
                else:
                    time.sleep(SLEEP_INTERVAL)
            while status == JOB_STATUS["RUNNING"]:
                clear_output(wait=True)
                output_job_tail(job_out_file, n=n, add_time=add_time, msg_type="i")
                status, _ = check_job_status(
                    job_id=job_id,
                    job_out_file=job_out_file,
                    job_err_file=job_err_file,
                    slurm=slurm,
                    verbose=-1,
                )
                time.sleep(0.1)
        elif 2 <= verbose < 4:
            waiting = True
            while waiting:
                if os.path.exists(job_out_file):
                    file_object = open(job_out_file)
                    waiting = False
                else:
                    time.sleep(SLEEP_INTERVAL)
            while status == JOB_STATUS["RUNNING"]:
                line = next(pytail_follow(file_object)).strip()
                if add_time:
                    message(
                        f"[{datetime.now().astimezone().strftime(TIME_FORMAT)}] {line}",
                        msg_type="info",
                    )
                else:
                    message(f"{line}", msg_type="info")
                status, _ = check_job_status(
                    job_id=job_id,
                    job_out_file=job_out_file,
                    job_err_file=job_err_file,
                    slurm=slurm,
                    verbose=-1,
                )
                time.sleep(SLEEP_INTERVAL)
        elif verbose >= 4:
            waiting = True
            while waiting:
                if os.path.exists(job_out_file):
                    file_object = open(job_out_file)
                    waiting = False
                else:
                    time.sleep(SLEEP_INTERVAL)
            while status == JOB_STATUS["RUNNING"]:
                line = next(pytail_follow(file_object)).strip()
                message(
                    f"[{datetime.now().astimezone().strftime(TIME_FORMAT)}] {line}",
                    msg_type="d",
                )
                time.sleep(SLEEP_INTERVAL)
        else:
            waiting = True
            while waiting:
                if os.path.exists(job_out_file):
                    waiting = False
                else:
                    time.sleep(SLEEP_INTERVAL)
            while status == JOB_STATUS["RUNNING"]:
                line = f"RUNNING {clock[i % 4]}"
                i += 1
                message(" " * 100, msg_type="i", end="\r", flush=True)
                message(
                    f"[{datetime.now().astimezone().strftime(TIME_FORMAT)}] {line}",
                    msg_type="i",
                    end="\r",
                    flush=True,
                )
                status, _ = check_job_status(
                    job_id=job_id,
                    job_out_file=job_out_file,
                    job_err_file=job_err_file,
                    slurm=slurm,
                    verbose=-1,
                )
                time.sleep(1 / 8)  # time.sleep(SLEEP_INTERVAL)
            message(" " * 100, msg_type="i", end="\r", flush=True)
            message(
                "[{}] {}".format(datetime.now().astimezone().strftime(TIME_FORMAT), "END"),
                msg_type="i",
                end="\n",
                flush=True,
            )
    except KeyboardInterrupt:
        if verbose >= 0:
            message(" " * 100, msg_type="w", end="\r")
            message("SIGINT signal detected!", msg_type="w")

    return None
