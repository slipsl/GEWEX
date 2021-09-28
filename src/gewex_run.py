#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==================================================================== #
# Author: Sonia Labetoulle                                             #
# Contact: sonia.labetoulle _at_ ipsl.jussieu.fr                       #
# Created: 2019                                                        #
# History:                                                             #
# Modification:                                                        #
# ==================================================================== #

# Standard library imports
# ========================
import psutil
# import os
from pathlib import Path
import datetime as dt
# from cftime import num2date, date2num
# import cftime as cf  # num2date, date2num
import pprint

# import numpy as np
# import pandas as pd
# from fortio import FortranFile
# from scipy.io import FortranFile
# from scipy.interpolate import interp1d
# from netCDF4 import Dataset

pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================
# import gewex_param as gw
# import gewex_netcdf as gnc


#######################################################################
def get_arguments():
  from argparse import ArgumentParser
  from argparse import RawTextHelpFormatter

  parser = ArgumentParser(
    formatter_class=RawTextHelpFormatter
  )
  # parser.add_argument("project", action="store",
  #                     help="Project name")
  # parser.add_argument("center", action="store",
  #                     help="Center name (idris/tgcc)")

  parser.add_argument(
    "runtype", action="store",
    type=int,
    choices=[1, 2, 3, 4, 5],
    help=(
      "Run type:\n"
      "  - 1 = AIRS / AM\n"
      "  - 2 = AIRS / PM\n"
      "  - 3 = IASI / AM\n"
      "  - 4 = IASI / PM\n"
      "  - 5 = Test mode (node = 0.0)\n"
    )
  )
  parser.add_argument(
    "date_start", action="store",
    type=lambda s: dt.datetime.strptime(s, "%Y%m%d"),
    help="Start date: YYYYMMJJ"
  )
  parser.add_argument(
    "date_end", action="store",
    type=lambda s: dt.datetime.strptime(s, "%Y%m%d"),
    help="End date: YYYYMMJJ"
  )

  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )
  parser.add_argument(
    "-f", "--force", action="store_true",
    help="If output files exsist, replace them."
  )

  parser.add_argument(
    "--notemp", action="store_true",
    help="Don't produce temperature files"
  )
  parser.add_argument(
    "--noh2o", action="store_true",
    help="Don't produce spec. humidity files"
  )
  parser.add_argument(
    "--nosurf", action="store_true",
    help="Don't produce surface type files"
  )

  # parser.add_argument("-d", "--dryrun", action="store_true",
  #                     help="only print what is to be done")
  return parser.parse_args()


#######################################################################

if __name__ == "__main__":

  run_deb = dt.datetime.now()

  # .. Initialization ..
  # ====================
  # ... Command line arguments ...
  # ------------------------------
  args = get_arguments()
  if args.verbose:
    print(args)

  # ... Constants ...
  # -----------------

  # ... Files and directories ...
  # -----------------------------
  project_dir = Path(__file__).resolve().parents[1]
  # dirin = project_dir.joinpath("input")
  # # dirin_3d = dirin.joinpath("AN_PL")
  # # dirin_2d = dirin.joinpath("AN_SF")
  dirout = Path("run")
  dirlog = dirout.joinpath("log")

  filebase = (
    F"gewex_r{args.runtype}_"
    F"{args.date_start:%Y%m%d}_{args.date_end:%Y%m%d}"
  )
  runfile = dirout.joinpath(F"{filebase}.sh")

  pgm = Path("src").joinpath("gewex_main2.py")
  conda_env = "gewex2"

  # print(pgm)
  # print(filebase)


  # .. Main program ..
  # ==================

  title = (
    F"GW{args.runtype}_"
    F"{args.date_start:%y}_"
    F"{args.date_start:%m%d}"
    F"{args.date_end:%m%d}"
  )
  # print(title)

  # run_args = [
  #   F"{args.runtype}",
  #   F"{args.date_start:%Y%m%d}",
  #   F"{args.date_end:%Y%m%d}",
  # ]

  run_opt1 = []
  if args.verbose:
    run_opt1.append("-v")
  if args.force:
    run_opt1.append("-f")

  run_opt2 = []
  if args.notemp:
    run_opt2.append("--notemp")
  if args.noh2o:
    run_opt2.append("--noh2o")
  if args.nosurf:
    run_opt2.append("--nosurf")

  # args_string = (
  #   "-" + "".join(run_opt1) +
  #   " " + " ".join(run_opt2) +
  #   " " + " ".join(run_args)
  # )
  

  string_list = [
    F"#!/bin/sh",
    F"",
    F"#PBS -N {title}",
    F"#PBS -q weeks2",
    F"#PBS -o {dirlog.joinpath(filebase)}.out",
    F"#PBS -e {dirlog.joinpath(filebase)}.err",
    F"#PBS -l mem=15gb,vmem=15gb",
    F"#PBS -l walltime=168:00:01",
    F"",
    F"date",
    F"",
    F"module purge",
    F"module load python/3.6-anaconda50",
    F"source activate {conda_env}",
    F"",
    F"cd $PBS_O_WORKDIR",
    F"",
    (
      F"python {pgm} "
      F"{' '.join(run_opt1)} "
      F"{' '.join(run_opt2)} "
      F"{args.runtype} "
      F"{args.date_start:%Y%m%d} {args.date_end:%Y%m%d}"
    ),
    F"",
    F"source deactivate",
    F"",
    F"date",
  ]

  with open(runfile, "w") as f:
    for line in string_list:
      f.write(F"{line}\n")

  print(F"\n{72*'='}\nRun ended in {dt.datetime.now() - run_deb}")

  exit()
