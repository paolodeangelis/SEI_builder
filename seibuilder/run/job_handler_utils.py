"""Additiona ussefule rutine for jobs handler."""
from datetime import datetime

from .._constants import TIME_FORMAT
from ..utils import message, pytail


def output_job_tail(stdout_file: str, n: int = 10, add_time: bool = True, msg_type: str = "info"):
    """Follow the STDOUT file and print the last line.

    Args:
        stdout_file (str): path to the STDOUT file.
        n (int, optional): number of line to print. Defaults to 10.
        add_time (bool, optional): show the date. Defaults to True.
        msg_type (str, optional):  type of message:
            -  'error' or 'e'     : errors messages
            -  'warning' or 'w'   : warning messages
            -  'info' or 'i'      : info messages
            -  'debug' or 'd'     : debug messages.
            Defaults to "info".
    """
    output = pytail(stdout_file, n=n)
    if isinstance(output, list):
        for output_ in output:
            if add_time:
                message(
                    f"[{datetime.now().astimezone().strftime(TIME_FORMAT)}] {output_}",
                    msg_type=msg_type,
                )
            else:
                message(f"{output_}", msg_type=msg_type)
    else:
        pass
