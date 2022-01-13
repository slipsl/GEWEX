#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
# ========================
import sys
from pathlib import Path
# import os.path
from math import floor, ceil
import datetime as dt
import numpy as np
from scipy.io import FortranFile
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import colors as clr
# from matplotlib.backends.backend_pdf import PdfPages
# from mpl_toolkits.basemap import Basemap
# from netCDF4 import Dataset

import pprint
pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================
sys.path.append(str(Path(__file__).resolve().parents[1].joinpath("src")))
import gwx_param as gwp
import gwx_variable as gwv
import gwx_netcdf as gwn


#######################################################################
def get_arguments():
  from argparse import ArgumentParser
  from argparse import RawTextHelpFormatter

  parser = ArgumentParser(
    formatter_class=RawTextHelpFormatter
  )

  parser.add_argument(
    "varname", action="store",
    # type=int,
    choices=["P", "T", "Q", "S"],
    help=(
      "Variable to check"
    )
  )
  parser.add_argument(
    "runtype", action="store",
    type=int,
    choices=[1, 2, 3, 4, 5, 6, 9],
    help=(
      "Run type:\n"
      "  - 1 = AIRS / AM\n"
      "  - 2 = AIRS / PM\n"
      "  - 3 = IASI / AM\n"
      "  - 4 = IASI / PM\n"
      "  - 5 = IASI-B / AM\n"
      "  - 6 = IASI-B / PM\n"
      "  - 9 = Test mode (node = 0.0)\n"
    )
  )
  parser.add_argument(
    "fileversion", action="store",
    help="File version"
  )

  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )

  return parser.parse_args()


#----------------------------------------------------------------------
def read_f77(variable, filein, grid):

  # print(variable.name)

  if variable.name == "S"
    dtype_in = ">i4"
    dtype_out = np.int32
  else:
    dtype_in = ">f4"
    dtype_out = np.float32

  try:
    with FortranFile(filein, "r", header_dtype=">u4") as f:
      rec = f.read_record(dtype=dtype_in).astype(dtype=dtype_out)
      # rec = f.read_record(dtype=">f4").astype(dtype=np.float32)
  except Exception as err:
    print(err)
    print(filein)
    var_out = None
    return var_out

  if variable.mode == "2d":
    shape = (grid.nlon, grid.nlat)
  else:
    shape = (grid.nlev + V.extralev, grid.nlon, grid.nlat)
    # if variable.name == "temp":
    #   shape = (grid.nlev+2, grid.nlon, grid.nlat)
    # else:
    #   shape = (grid.nlev, grid.nlon, grid.nlat)

  var_out = rec.reshape(shape)
  var_out = np.rollaxis(var_out, -1, -2)

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

  # ... Constants ...
  # -----------------
  fileversion = args.fileversion

  # ... Files and directories ...
  # -----------------------------
  project_dir = Path(__file__).resolve().parents[1]

  # .. Main program ..
  # ==================

  # ... Initialize things ...
  # -------------------------
  instru = gwp.InstruParam(args.runtype)
  params = gwp.GewexParam(project_dir)

  if args.verbose:
    print(instru)
    print(params)

  # Grids
  tggrid = gwv.TGGrid()
  tggrid.load()

  # Variables
  V = gwv.VarOut(args.varname, instru, fileversion)

  print(V.longname, instru.name, instru.ampm)
  print(
    F"{20*' '}   min  "
    F"   max  "
    F"  mean  "
    F"   std  "
    F"     neg  "
    F"  zeroes  "
    F"     pos  "
  )

  pattern = F"**/unmasked_ERA5_{instru.name}_{V.outstr}.*.{instru.ampm}_{fileversion}"
  for f in sorted(params.dirout.glob(pattern)):
    V.pyvalues = read_f77(V, f, tggrid)
    if V.pyvalues is None :
      continue

    (nb_zeroes, nb_neg, nb_pos) = (
      np.count_nonzero(cond)
        for cond in ((V.pyvalues == 0.), (V.pyvalues < 0.), (V.pyvalues > 0.))
    )

    print(
      F"{''.join(f.suffixes)} : "
      F"{V.pyvalues.min():6.2f}  "
      F"{V.pyvalues.max():6.2f}  "
      F"{V.pyvalues.mean():6.2f}  "
      F"{V.pyvalues.std():6.2f}  "
      F"{nb_neg:8d}  "
      F"{nb_zeroes:8d}  "
      F"{nb_pos:8d}  "
    )

  print(F"\n{72*'='}\nRun ended in {dt.datetime.now() - run_deb}")

  exit()
