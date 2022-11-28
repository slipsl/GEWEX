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
from pathlib import Path
import datetime as dt
from scipy.io import FortranFile
import numpy as np

# Application library imports
# ========================
import gewex_param as gwp
import gewex_variable as gwv
# import gewex_netcdf as gwn


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
    "varname", action="store",
    # type=int,
    choices=["Psurf", "temp", "h2o"],
    help=(
      "Variable to plot: \"Psurf\", \"temp\", \"h2o\""
    )
  )
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
    "date", action="store",
    type=lambda s: dt.datetime.strptime(s, "%Y%m%d"),
    help="Date: YYYYMMJJ"
  )

  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )
  # parser.add_argument("-d", "--dryrun", action="store_true",
  #                     help="only print what is to be done")
  return parser.parse_args()


#----------------------------------------------------------------------
def read_f77(variable, filein, grid):

  with FortranFile(filein, "r", header_dtype=">u4") as f:
    rec = f.read_record(dtype=">f4").astype(dtype=np.float32)

  if variable.mode == "2d":
    shape = (grid.nlon, grid.nlat)
  else:
    if variable.name == "temp":
      shape = (grid.nlev+2, grid.nlon, grid.nlat)
    else:
      shape = (grid.nlev, grid.nlon, grid.nlat)

  var_out = rec.reshape(shape)

  return var_out


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

  # ... Files and directories ...
  # -----------------------------
  project_dir = Path(__file__).resolve().parents[1]

  # ... Appli parameters ...
  # ------------------------
  instru = gwp.InstruParam(args.runtype)
  params = gwp.GewexParam(project_dir)

  # ... Load NetCDF & target grids ...
  # ----------------------------------
  # ncgrid = gwn.NCGrid()
  # ncgrid.load(variable.ncfiles)
  tggrid = gwv.TGGrid()
  tggrid.load()


  if args.verbose:
    print(instru)
    print(params)

  variable = gwv.Variable(args.varname, instru)
  if args.verbose:
    print(variable)
  variable.fileversion = "SL2"


  filein = variable.pathout(params.dirout, args.date)
  fileout = Path(F"{filein}.dat")

  print(filein)
  print(fileout)
  print(type(filein))
  print(type(fileout))

  variable.pyvalues = \
      read_f77(variable, filein, tggrid)
  print(variable.pyvalues.shape)


  with open(fileout, "w") as f:
    for row in variable.pyvalues:
      # print(row)
      line = " ".join((F"{i:8.2f}" for i in row))
      f.write(F"{line}\n")



  print(F"\n{72*'='}\nRun ended in {dt.datetime.now() - run_deb}")

  exit()
