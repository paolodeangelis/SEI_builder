#!/usr/bin/env python3
"""Script to set `Material Project` API Key."""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath("."))


def load_mpinterfaces():  # noqa: D103
    print("[I]: Loading mpinteface (may take a  while)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from seibuilder.mpinterfaces import PACKAGE_PATH
    global MPINT_CONFIG_YAML
    MPINT_CONFIG_YAML = os.path.join(PACKAGE_PATH, "mpint_config.yaml")


def check_and_remove(mpi_config_yaml):  # noqa: D103
    if os.path.exists(mpi_config_yaml):
        os.remove(mpi_config_yaml)
        print("\033[33m[W]: the configuration file already exist, it will be overwrite\033[0m")


def write_yaml(mpi_config_yaml, api_key=None):  # noqa: D103
    if api_key is None:
        api_key = "null"
    with open(mpi_config_yaml, "w") as file_:
        file_.write(
            f"""username: null  # your UF HPC username
mp_api: {api_key}  # your Materials Project API key
normal_binary: null # /path/to/std/vasp/executable
twod_binary: null  # /path/to/2D/vasp/executable
vdw_kernel: null   # /path/to/vdw_kernel.bindat  (leave as null if the kernel is hard-coded into VASP)
potentials: null  # /path/to/POTCAR/files
queue_system: slurm  # Change to pbs if on a PBS system
queue_template: config_files/ # path/to/queue/template containing account info, processor config 'submit_script'
                """
        )


def main(api_key=None):  # noqa: D103
    load_mpinterfaces()
    check_and_remove(MPINT_CONFIG_YAML)
    write_yaml(MPINT_CONFIG_YAML, api_key=api_key)


if __name__ == "__main__":
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-k", "--apikey", required=True, help="Material Project API key")
    args = vars(ap.parse_args())
    main(api_key=args["apikey"])
    print(f"[I]: mpinterface configuration file saved in: \n\t{MPINT_CONFIG_YAML}")
