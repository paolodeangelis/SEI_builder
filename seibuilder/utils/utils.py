"""Set of util functions."""
import os
import shutil
import time
from datetime import datetime
from io import TextIOWrapper
from typing import Union

from .._constants import SLEEP_INTERVAL, TIME_FORMAT


def message(msg: str, msg_type: str = "info", add_date: bool = False, **kwargs):
    """Print on screen useful messages.

    Args:
        msg (str): message to print
        msg_type (str, optional): type of message
            -  'error' or 'e'     : errors messages
            -  'warning' or 'w'   : warning messages
            -  'info' or 'i'      : info messages
            -  'debug' or 'd'     : debug messages. Defaults to "info".
        add_date (bool, optional): if True show the time. Defaults to False.

    Kwargs:
        **kwargs: other ``print`` function kwargs
    """
    if add_date:
        msg = f"[{datetime.now().astimezone().strftime(TIME_FORMAT)}] {msg}"

    if msg_type.lower() == "debug" or msg_type.lower() == "d":
        print("\x1b[34m[D]: " + msg + "\x1b[0m", **kwargs)
    elif msg_type.lower() == "info" or msg_type.lower() == "i":
        print("[I]: " + msg, **kwargs)
    elif msg_type.lower() == "warning" or msg_type.lower() == "w":
        print("\x1b[33m[W]: " + msg + "\x1b[0m", **kwargs)
    elif msg_type.lower() == "error" or msg_type.lower() == "e":
        print("\x1b[31m[E]: " + msg + "\x1b[0m", **kwargs)


def makedir(path: str, verbose: int = 1):
    """Make a folder function.

    Args:
        path (str): directory path.
        verbose (int, optional): loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages.
            Defaults to 1.
    """
    path = path.split(os.sep)
    for i in range(len(path)):
        path_ = os.sep.join(path[: i + 1])
        if path_ == "":
            continue
        if not os.path.isdir(path_):
            os.mkdir(path_)
            if verbose >= 2:
                message(f"Folder {path_} maked", msg_type="i")
        else:
            if verbose >= 1 and i + 1 == len(path):
                message(f"Folder {path_} already exist", msg_type="w")


def pytail(file_path: str, n: int = 10, encoding: str = "utf8") -> Union[bool, list]:
    """Print last n lines from the file_path.

    Args:
        file_path (str): File path.
        n (int): Optional; number of last line to read.
        encoding (str): Optional; character encoding (see: https://docs.python.org/3/howto/unicode.html)

    Returns:
        Union[bool, list]: muliple possible output:
            -  bool: False the file is empty
            -  list: a list with the last lines.
    """
    avg_line_len = 100
    string = b""
    try:
        with open(file_path, "rb") as f:
            try:
                f.seek(0, 2)
                while string.count(b"\n") < n:
                    cursor = f.tell()
                    f.seek(cursor - avg_line_len * n)
                    string = f.read(avg_line_len * n) + string
                    f.seek(cursor - avg_line_len * n)
            except Exception:  # noqa: 722
                f.seek(0, 0)
                string = f.read()
            out_b = string.split(b"\n")
            rows = len(out_b)
            if rows >= n:
                start = rows - n
            else:
                start = 0
            out = []
            for i in range(start, rows):
                out += [out_b[i].decode(encoding)]
            return out
    except Exception:  # noqa: 722
        return False


def pytail_follow(file_object: TextIOWrapper) -> str:
    """Recursive function to read and print last lines of a file.

    Args:
        file_object (TextIOWrapper): file buffered text stream.

    Yields:
        str: file line.
    """
    while True:
        line = file_object.readline()
        if line:
            yield line
        else:
            where = file_object.tell()
            line = file_object.readline()
            if not line:
                time.sleep(SLEEP_INTERVAL)
                file_object.seek(where)
            else:
                yield line


# def get_address():
#     ip_public = requests.get("http://ip.42.pl/raw").text
#     address, _, _ = socket.gethostbyaddr(ip_public)
#     return {"ip": ip_public, "address": address}


def _progressbar(percentage: float, info: str = "", screen: int = 100, status: str = "info"):
    if percentage is None:
        percentage = 0.0
    if info != "":
        info = info.strip() + " "
    bar_length = screen - len(info) - 2 - 6
    status_chars = [
        " ",
        "\u258f",
        "\u258e",
        "\u258d",
        "\u258c",
        "\u258b",
        "\u258a",
        "\u2589",
        "\u2588",
    ]
    # if percentage <= 1.:
    #     percentage *= 100
    length = percentage * bar_length / 100
    units = int(length // 1)  # floor of the percentage
    decimal = length % 1  # mantissa of the percentage
    empty = bar_length - units
    full = 1 - int(percentage / 100)
    if status == "success":
        color = "\x1b[32m"
    elif status == "warning":
        color = "\x1b[33m"
    elif status == "danger":
        color = "\x1b[31m"
    else:
        color = "\x1b[0m"
    # Print bar
    text = "{:s}{:s}{:s}\x1b[0m {:4.1f}%".format(
        info,
        color,
        "\u2588" * units + status_chars[int(decimal * 8)] * full + " " * empty,
        percentage,
    )
    return text


def copy_file(source: str, destination: str, verbose: int = 0):
    """Copy a file in a new destination.

    Args:
        source (str): file to copy.
        destination (str): where to copy.
        verbose (int, optional):  loudness controller:
            -  0: print errors
            -  1: print errors and warnings
            -  2: print errors, warnings and info
            -  3: print errors, warnings, info and debugger messages.
            Defaults to 0.

    Raises:
        ValueError: if source file does not exist.
    """
    if os.path.exists(source):
        if os.path.exists(destination):
            os.remove(destination)
            shutil.copy2(source, destination)
            if verbose >= 1:
                message(f"File {destination} already existed, we overwrite it", msg_type="w")
        else:
            shutil.copy2(source, destination)
            if verbose >= 2:
                message(f"File copied in {destination}", msg_type="i")
    else:
        if verbose >= 0:
            message(f"Was impossible to find the source file {source}", msg_type="e")
        raise ValueError(f"Was impossible to find the source file {source}")


def are_we_in_a_notebook() -> bool:
    """Check in which environment we are.

    Returns:
        bool: booleand option:
            -  True: if the script is running inside a Jupyter notebook.
            -  False: if is running in Python or IPython console.
    """
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
