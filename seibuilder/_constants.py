import os

SEP = "/"
SLEEP_INTERVAL = 0.05
TIMEOUT = 10  # [s]

WORK_DIR = os.getcwd()
# LOGO = __logo__

USER_FILE = "user.json"
ERR_FILE = "output-{:s}-%j.err"
OUT_FILE = "output-{:s}-%j.out"
DATE_FORMAT = "%d/%m/%Y %H:%M:%S %Z"
TIME_FORMAT = "%H:%M:%S %Z"
WRAPPER_MARK = "LAMMPS_WRAPPER_MARK"
JOB_STATUS = {
    "BOOT_FAIL": "BF",
    "CANCELLED": "CA",
    "COMPLETED": "CD",
    "CONFIGURING": "CF",
    "COMPLETING": "CG",
    "DEADLINE": "DL",
    "FAILED": "F",
    "OUT_OF_MEMORY": "OOM",
    "PENDING": "PD",
    "PREEMPTED": "PR",
    "RUNNING": "R",
    "RESV_DEL_HOLD": "RD",
    "REQUEUE_FED": "RF",
    "REQUEUE_HOLD": "RH",
    "REQUEUED": "RQ",
    "RESIZING": "RS",
    "REVOKED": "RV",
    "SIGNALING": "SI",
    "SPECIAL_EXIT": "SE",
    "STAGE_OUT": "SO",
    "STOPPED": "ST",
    "SUSPENDED": "S",
    "TIMEOUT": "TO",
    "NO_READY": "NR",
    "READY": "RE",
}
JOB_STATUS_CONV = {
    "D": JOB_STATUS["SUSPENDED"],
    "I": JOB_STATUS["PENDING"],
    "R": JOB_STATUS["RUNNING"],
    "S": JOB_STATUS["SUSPENDED"],
    "T": JOB_STATUS["CANCELLED"],
    "Z": JOB_STATUS["STOPPED"],
}
JOB_STATUS_INV = {v: k for k, v in JOB_STATUS.items()}
