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

from netCDF4 import Dataset
import numpy as np
# from scipy.io import FortranFile

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
    help="Input NetCDF file"
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

  with Dataset(args.filein, "r", format="NETCDF4") as f_in:

    print(F"{72*'='}")
    print(
      F"Informations for file:\n"
      F"{Path(args.filein).name}"
    )

    print(F"\n\n{72*'='}")
    print(F"Attributes:")
    print(72*"-")
    for aname in f_in.ncattrs():
      print(F"{aname}: {f_in.getncattr(aname)}")

    print(F"\n\n{72*'='}")
    ndim = len(f_in.dimensions)
    print(F"{ndim} dimensions:")
    print(72*"-")

    for dname, dim in f_in.dimensions.items():
      units = f_in.variables[dname].units
      val = f_in.variables[dname][...]

      if "time" in dname:
        calendar = f_in.variables[dname].calendar
        vmin = num2date(val.min(), calendar, units)
        vmax = num2date(val.max(), calendar, units)
        punits = ""
      else:
        vmin, vmax = (val.min(), val.max())
        punits = f_in.variables[dname].units

      print(F"- {dname}, size = {dim.size}, units = {units}")
      string = (
        F"  "
        F"min = {vmin} {punits} ; "
        F"max = {vmax} {punits}"
      )
      print(string)
      print(72*"-")

    print(F"\n\n{72*'='}")
    print("Variables:")
    print(72*"-")

    nsamples = 10

    for vname, var in f_in.variables.items():
      if vname not in f_in.dimensions:

        print(F"- {vname}, shape = {var.shape}")

        for ncattr in var.ncattrs():
          print(F"  {ncattr}: {f_in.variables[vname].getncattr(ncattr)}")
        print(72*"-")

        print("Sample values:")
        for i in range(nsamples):
          # print(F"i = {i}")
          indices = tuple(random.randrange(plage) for plage in var.shape)
          str_indices = ", ".join(
            F"{str(i):>3s}" for i in indices
          )
          print(
            F"{var.name}[{str_indices}] = "
            F"{var[indices]}"
          )

        answer = input(
          F"\nPrint basic statistics for {var.name}?\n"
          F"  WARNING : this can take a long time for profile variables\n"
          F"(y/N) : "
        ) or "N"

        if answer.lower() == "y":
          vmin = np.inf
          vmax = -np.inf
          vsum = 0
          for t in range(var.shape[0]):
            sub_var = var[t, ...]
            # print(sub_var[...].min(), sub_var[...].max(), )
            vmin = min(vmin, sub_var[...].min())
            vmax = max(vmax, sub_var[...].max())
            vsum = vsum + sub_var.sum()
            # print(F" => {vmin} ; {vmax} ; {sub_var.mean()} ; {vsum}")

            if not (t % (24 * 3)):
              print(
                F"Partial results ({t}/{var.shape[0]}):\n"
                F"- min({var.name})  = {sub_var.min()}\n"
                F"- max({var.name})  = {sub_var.max()}\n"
                F"- mean({var.name}) = {sub_var.mean()}\n"
                F"- std({var.name})  = {sub_var.std()}"
              )

          print(
            F"\nGlobal results:\n{15*'~'}\n"
            F"- min({var.name})  = {vmin}\n"
            F"- max({var.name})  = {vmax}\n"
            F"- mean({var.name}) = {vsum / var.size}"
          )

  print(F"\n{72*'='}\nRun ended in {dt.datetime.now() - run_deb}")

  exit()
