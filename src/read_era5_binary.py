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
from pathlib import Path
import datetime as dt
import cftime as cf  # num2date, date2num
import random
import pprint

# from netCDF4 import Dataset
import numpy as np
from scipy.io import FortranFile

pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================


#######################################################################
def get_arguments():
  from argparse import ArgumentParser
  from argparse import RawTextHelpFormatter

  parser = ArgumentParser(
    formatter_class=RawTextHelpFormatter
  )

  parser.add_argument(
    "filein", action="store",
    help="Input binary file"
  )

  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )

  return parser.parse_args()


#----------------------------------------------------------------------
def num2date(val, calendar, units):

  return cf.num2date(
    val,
    units=units,
    calendar=calendar,
    only_use_cftime_datetimes=False,
    only_use_python_datetimes=True,
    has_year_zero=None
  )


#----------------------------------------------------------------------
def get_dimensions(filein):

  (nlon, nlat) = (1440, 721)

  if "L2_P_surf_daily_average" in filein:
    nlev = None
    dtype_in = ">f4"
    dtype_out = np.float32
  elif "L2_H2O_daily_average" in filein:
    nlev = 23
    dtype_in = ">f4"
    dtype_out = np.float32
  elif "L2_temperature_daily_average" in filein:
    nlev = 25
    dtype_in = ">f4"
    dtype_out = np.float32
  elif "L2_SurfType" in filein:
    nlev = None
    dtype_in = ">i4"
    dtype_out = np.int32
  elif "L2_status" in filein:
    nlev = None
    dtype_in = ">i4"
    dtype_out = np.int32

  return nlon, nlat, nlev, dtype_in, dtype_out


#----------------------------------------------------------------------
def read_f77(filein, nlon, nlat, nlev, dtype_in, dtype_out):

  with FortranFile(filein, "r", header_dtype=">u4") as f:
    rec = f.read_record(dtype=dtype_in).astype(dtype=dtype_out)

  if nlev:
    shape = (nlev, nlon, nlat)
  else:
    shape = (nlon, nlat)

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

  # ... Files and directories ...
  # -----------------------------

  # .. Main program ..
  # ==================


  (nlon, nlat, nlev, dtype_in, dtype_out) = get_dimensions(args.filein)
  print(nlon, nlat, nlev, dtype_in, dtype_out)

  var = read_f77(args.filein, nlon, nlat, nlev, dtype_in, dtype_out)
  print(var.shape)

  print(F"{72*'='}")
  print(
    F"Informations for file:\n"
    F"{Path(args.filein).name}"
  )

  if nlev:
    ndim = 3
  else:
    ndim = 2
  print(F"\n\n{72*'='}")
  print(F"{ndim} dimensions:")
  print(72*"-")

  for dname, dim in (("nlev", nlev), ("nlat", nlat), ("nlon", nlon), ):
    print(F"- {dname}, size = {dim}")

  nsamples = 10
  print(F"\n\n{72*'='}")
  print("Sample values:")
  print(72*"-")
  for i in range(nsamples):
    # print(F"i = {i}")
    indices = tuple(random.randrange(plage) for plage in var.shape)
    str_indices = ", ".join(
      F"{str(i):>3s}" for i in indices
    )
    print(
      F"- var[{str_indices}] = "
      F"{var[indices]}"
    )

  print(F"\n\n{72*'='}")
  print(F"Basic statistics")
  print(72*"-")
  print(
    F"- min  = {var.min()}\n"
    F"- max  = {var.max()}\n"
    F"- mean = {var.mean()}\n"
    F"- std  = {var.std()}"
  )

  print(F"\n{72*'='}\nRun ended in {dt.datetime.now() - run_deb}")

  exit()
