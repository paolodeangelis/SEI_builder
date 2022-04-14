"""Job handler functions using ``bash`` scripts."""
import os
from datetime import datetime
from subprocess import PIPE, Popen
from typing import Tuple

from .._constants import (
    DATE_FORMAT,
    ERR_FILE,
    JOB_STATUS,
    JOB_STATUS_CONV,
    OUT_FILE,
    WRAPPER_MARK,
)
from ..utils import message, pytail


class BashError(RuntimeError):
    """Bash error class."""


def write_script_bash(
    cmd: str or list,
    sim_id: str,
    job_settings: str = None,
    script_path: str = None,
    verbose: int = 0,
) -> str:
    """Write the ``bash`` script file.

    Args:
        cmd (strorlist): job commands.
        sim_id (str): simulation id.
        job_settings (str, optional): jobs settings file path. Defaults to None.
        script_path (str, optional): path where to save the ``bash`` script. Defaults to None.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages.
            Defaults to 0.

    Returns:
        str: script file path.
    """
    if script_path is None:
        script_path = f"run_{sim_id}.sh"
    # load JOB settings:
    # job_settings = load_json(job_settings)

    with open(script_path, "w") as outfile:
        outfile.write("#!/bin/bash" + "\n")
        outfile.write(f"#ID {sim_id}" + "\n")
        try:
            outfile.write(f"#USER {os.getlogin()}" + "\n")
        except OSError:
            outfile.write("#USER unknow" + "\n")

        # MODULE
        # if "module" in job_settings:
        #     outfile.write("\n" + "# MODULEs" + "\n")
        #     for key, value in job_settings["module"].items():
        #         if isinstance(value, list):
        #             for value_ in value:
        #                 outfile.write(f"module {key} {value_}" + "\n")
        #         else:
        #             outfile.write(f"module {key} {value}" + "\n")
        # OTHER
        # if "other" in job_settings:
        #     outfile.write("\n" + "# OTHERs" + "\n")
        #     for key, value in job_settings["other"].items():
        #         if isinstance(value, list):
        #             for value_ in value:
        #                 outfile.write(f"{key} {value_}" + "\n")
        #         else:
        #             outfile.write(f"{key} {value}" + "\n")
        # COMMANDS
        outfile.write("\n" + "# COMMANDS" + "\n")
        if isinstance(cmd, list):
            cmd = "\n".join(cmd)
            outfile.write(cmd + "\n")
        else:
            outfile.write(cmd + "\n")
        # END MARK
        outfile.write("\n" + "# LAMMPS WRAPPER MARK" + "\n")
        outfile.write("END=$? \n")
        outfile.write(f'NOW=$(date +"{DATE_FORMAT:s}") \n')
        outfile.write(
            "if [ $END -eq 0 ]; then\n"
            + f"    echo '{WRAPPER_MARK};COMPLETED;'$NOW\n"
            + "else\n"
            + f"    echo '{WRAPPER_MARK};FAILED;'$NOW\n"
            + "fi\n"
        )
        if verbose >= 2:
            message(f"Script save in {script_path}", msg_type="i")
    return script_path


def submit_job_bash(script_path: str, sim_id: str, afterok: int = None, verbose: int = 0) -> Tuple[int, str, str]:
    """Submit the job in background.

    Args:
        script_path (str): the job input script.
        sim_id (str): simulation id.
        afterok (int, optional): [PlaceHolder] depended job ID to start after. Defaults to None.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages.
            Defaults to 0.

    Raises:
        BashError: raise error if somethin goes wrong whit starting the job.

    Returns:
        Tuple[int, str, str]: tuple containing:
            -  int: the command PID (equivalent to job ID).
            -  str: the path where the job ``STDOUT`` is stored.
            -  str: the path where the job ``STDERR`` is stored.
    """
    job_out_file = OUT_FILE.format(sim_id).replace("-%j", "")
    job_err_file = ERR_FILE.format(sim_id).replace("-%j", "")
    pid = None
    with open("tmp.sh", "w") as file:
        file.write("#!/bin/bash" + "\n")
        cmd = ["bash", script_path, "1>", job_out_file, "2>", job_err_file, "&"]
        if afterok is not None:
            add_cmd = [
                f"prev_pid={afterok}\n"
                + "echo 'Waithing the end of previus simulation (PID=$prev_pid)'\n"
                + "wait $prev_pid\n"
                + "echo 'Simulation (PID=$prev_pid) ends!'\n"
            ]
            cmd = add_cmd + cmd
            if verbose >= 3:
                message("Add waiting commands to submission script!", msg_type="d")
        file.write(" ".join(cmd) + "\n")
        file.write("echo $!" + "\n")
    proc = Popen(["bash", "tmp.sh"], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    pid = int(out.decode("ascii").split("\n")[0])
    if pid is None:
        if verbose >= 0:
            message(f"Was impossible submit the job {script_path}", msg_type="e")
        raise BashError(f"Was impossible submit the job {script_path}")
    os.remove("tmp.sh")
    return pid, job_out_file, job_err_file


def check_out_file_mark(job_out_file: str, job_err_file: str, verbose: int = 0) -> Tuple[str, datetime]:
    """Check the job status by looking at the ``STDOUT`` and ``STDERR`` files.

    Args:
        job_out_file (str): the path where the job ``STDOUT`` is stored.
        job_err_file (str): the path where the job ``STDERR`` is stored.
        verbose (int, optional): loudness controller:
                -  0: print errors
                -  1: print errors and warnings
                -  2: print errors, warnings and info
                -  3: print errors, warnings, info and debugger messages.
                Defaults to 0.

    Raises:
        ValueError: if STDOUT file not found.
        ValueError: if STDERR file not found.

    Returns:
        Tuple[str, datetime]: tuple containing:
            -  str: job status.
            -  datetime: the ending time if completed, None otherwise.
    """
    status = None
    end_date = None
    if not os.path.exists(job_out_file):
        if verbose >= 0:
            message(
                f"Cannot find the STDOUT file {os.path.abspath(job_out_file)}",
                msg_type="e",
            )
        raise ValueError(f"Cannot find the STDOUT file {os.path.abspath(job_out_file)}")
    else:
        out = pytail(job_out_file)
        if out:
            for i in range(len(out) - 1, -1, -1):
                if WRAPPER_MARK in out[i]:
                    if "COMPLETED" in out[i]:
                        status = JOB_STATUS["COMPLETED"]
                    else:
                        status = JOB_STATUS["FAILED"]
                    end_date = datetime.strptime(out[i].split(";")[-1], DATE_FORMAT).astimezone()
        if status is None:
            if not os.path.exists(job_err_file):
                if verbose >= 0:
                    message(
                        f"Cannot find the STDERR file {os.path.abspath(job_err_file)}",
                        msg_type="e",
                    )
                raise ValueError(f"Cannot find the STDERR file {os.path.abspath(job_err_file)}")
            if pytail(job_err_file):
                status = JOB_STATUS["FAILED"]
            else:
                status = JOB_STATUS["STOPPED"]
    return status, end_date


def check_job_status_bash(pid: int, job_out_file: str, job_err_file: str, verbose: int = 0) -> Tuple[str, datetime]:
    """Check job status ``bash``.

    Args:
        pid (int): the command PID (equivalent to job ID).
        job_out_file (str): the path where the job `STDOUT` is stored.
        job_err_file (str): the path where the job `STDERR` is stored.
        verbose (int, optional): loudness controller:
                -  0: print errors
                -  1: print errors and warnings
                -  2: print errors, warnings and info
                -  3: print errors, warnings, info and debugger messages.
                Defaults to 0.

    Returns:
        Tuple[str, datetime]: tuple containing:
            -  str: job status.
            -  datetime: the ending time if completed, None otherwise.
    """
    status = None
    end_date = None
    proc = Popen(["ps", f"{pid}"], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode("utf8").split("\n")
    if verbose >= 3:
        message(f"ps {pid}", msg_type="d")
        for out_ in out:
            message(f"\t{out_}", msg_type="d")
    try:
        if verbose >= 3:
            message("Check the job stautus", msg_type="d")
        status_key = out[1].split()[2][0]
        status = JOB_STATUS_CONV[status_key]
        if verbose >= 3:
            message(f"status {status_key} = {JOB_STATUS_CONV[status_key]}", msg_type="d")
    except:  # noqa: E722
        if verbose >= 3:
            message(
                f"Check the job output files {os.path.abspath(job_out_file)}",
                msg_type="d",
            )
        status, end_date = check_out_file_mark(job_out_file, job_err_file)
    return status, end_date
