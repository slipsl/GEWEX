#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==================================================================== #
# Author: Sonia Labetoulle                                             #
# Contact: sonia.labetoulle _at_ ipsl.jussieu.fr                       #
# Created: 2019                                                        #
# History:                                                             #
# Modification:                                                        #
# ==================================================================== #

# This must come first
from __future__ import print_function, unicode_literals, division

# Standard library imports
# ========================
import os
import datetime as dt
# from cftime import num2date, date2num
import pprint


import numpy as np
import pandas as pd
from fortio import FortranFile
from netCDF4 import Dataset

pp = pprint.PrettyPrinter(indent=2)


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
    choices=[1, 2, 3, 4],
    help= "Run type:\n"
          "  - 1 = AIRS / AM\n"
          "  - 2 = AIRS / PM\n"
          "  - 3 = IASI / AM\n"
          "  - 4 = IASI / PM\n"
  )
  parser.add_argument(
    "date_start", action="store",
    type=lambda s: dt.datetime.strptime(s, '%Y%m%d'),
    help="Start date: YYYYMMJJ"
  )
  parser.add_argument(
    "date_end", action="store",
    type=lambda s: dt.datetime.strptime(s, '%Y%m%d'),
    help="End date: YYYYMMJJ"
  )

  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="verbose mode"
  )

  # parser.add_argument("-d", "--dryrun", action="store_true",
  #                     help="only print what is to be done")
  return parser.parse_args()


#----------------------------------------------------------------------
def get_filein(varname, date_curr):

  # if varname == "ta" or varname == "q":
  if varname in pl_vars:
    vartype = "ap1e5"
    pathin = dirin_pl
  # elif varname == "sp" or varname == "skt":
  elif varname in sf_vars:
    vartype = "as1e5"
    pathin = dirin_sf

  # yyyymm = dt.datetime.strftime(date_curr, "%Y%m")

  # return F"{varname}.{yyyymm}.{vartype}.GLOBAL_025.nc"
  return (
    os.path.join(
      pathin,
      F"{date_curr:%Y}",
      F"{varname}.{date_curr:%Y%m}.{vartype}.GLOBAL_025.nc",
    )
  )

  # ta.202102.ap1e5.GLOBAL_025.nc
  # q.202102.ap1e5.GLOBAL_025.nc

  # skt.202102.as1e5.GLOBAL_025.nc
  # sp.202102.as1e5.GLOBAL_025.nc


#----------------------------------------------------------------------
def get_fileout(varname, date_curr):

  # F"unmasked_ERA5_AIRS_V6_L2_H2O_daily_average.20080220.PM_05"

  return (
    F"unmasked_ERA5_{instrument}_{varname}.{date_curr:%Y%m%d}.{ampm}_{fileversion}"
  )

  # ta.202102.ap1e5.GLOBAL_025.nc
  # q.202102.ap1e5.GLOBAL_025.nc

  # skt.202102.as1e5.GLOBAL_025.nc
  # sp.202102.as1e5.GLOBAL_025.nc


#----------------------------------------------------------------------
def lon_time(lon, lt_instru):

  l = lon
  if lon > 180.:
    l = lon - 360.
  l = l / 15.

  # univT = [i - 24. + 0.5 for i in range(72)]
  univT = [i - 24. for i in range(72)]
  # print(univT)
  localT = [i + l for i in univT]
  # print(localT)
  deltaT = [abs(i - lt_instru) for i in localT]
  # print(deltaT)


  print(
    " TU     TL     dT      "
    " TU     TL     dT      "
    " TU     TL     dT"
  )
  for i in range(24):
    print(
      F"{univT[i]:6.2f} {localT[i]:6.2f} {deltaT[i]:6.2f}   "
      F"{univT[i+24]:6.2f} {localT[i+24]:6.2f} {deltaT[i+24]:6.2f}   "
      F"{univT[i+48]:6.2f} {localT[i+48]:6.2f} {deltaT[i+48]:6.2f}   "
    )


  (imin1, imin2) = np.argsort(deltaT)[0:2]

  w1 = deltaT[imin1] / (deltaT[imin1] + deltaT[imin2])
  w2 = deltaT[imin2] / (deltaT[imin1] + deltaT[imin2])

  return (imin1, imin2, w1, w2)

#----------------------------------------------------------------------
def read_netcdf(filename):

  print(F"Reading {filename}\n"+80*"=")

  with Dataset(filename, "r", format="NETCDF4") as f_in:
    # print(f_in.data_model)
    # print(f_in.dimensions)

    # for obj in f_in.dimensions.values():
    #   print(obj)

    for obj in f_in.variables.values():
      print(obj)

    nc_time = f_in.variables["time"]
    nc_lon  = f_in.variables["longitude"]

    offsets = [
      (date_curr.day - i) * 24 for i in [2, 1, 0]
    ]
    print(offsets)
    nb_steps = 24

    times = []
    for o in offsets:
      times.extend(nc_time[o:o+nb_steps])
    for t in times:
      print(dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(t)))
      # num2date(times[:],units=times.units,calendar=times.calendar)

    # for time in nc_time:
    #   print(
    #     dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(time))
    #   )

    lt_instru = 13.5
    offset = (date_curr.day - 1) * 24

    print(80*"-")
    for lon in nc_lon[::180]:
      idx_t1, idx_t2, w_t1, w_t2 = lon_time(lon, lt_instru)

      tl = date_curr + dt.timedelta(hours=lon/15.)
      t_instru = date_curr + dt.timedelta(hours=lt_instru)
      t_cible = t_instru - dt.timedelta(hours=lon/15.)

      print(
        F"lon: {lon} => "
        F"tl = {tl}, "
        # F"tu = {lt_instru - lon/15.} "
        F"tu = {t_cible} "
        F"(lt_instru = {lt_instru})"
      )

      t_ori = dt.datetime(1800, 1, 1)
      t1 = t_ori + dt.timedelta(hours=float(times[idx_t1]))
      t2 = t_ori + dt.timedelta(hours=float(times[idx_t2]))
      # t1 = t_ori + dt.timedelta(hours=float(times[idx_t1 + offset]))
      # t2 = t_ori + dt.timedelta(hours=float(times[idx_t2 + offset]))

      print(
        F" - t1: {t1:%b %d %Hh} - Weight: {w_t1}\n"
        F" - t2: {t2:%b %d %Hh} - Weight: {w_t2}"
      )
      print(80*"-")

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

  # print(type(args.date_start))
  # print(type(args.date_end))

  # date_start = dt.datetime.strptime(args.date_start, "%Y%m%d")
  # date_end = dt.datetime.strptime(args.date_end, "%Y%m%d")

  # print("year:  ", args.date_start.year)
  # print("month: ", args.date_start.month)

  # ... Constants ...
  # -----------------
  if args.runtype == 1 or args.runtype == 2 :
    instrument = "AIRS_V6"
    coeff_h2o = 1000.0
    if args.runtype == 1:
      lt_instru = 1.5
      ampm = "AM"
    else:
      lt_instru = 13.5
      ampm = "PM"
  else:
    instrument = "IASI"
    coeff_h2o = 1.0
    if args.runtype == 3:
      lt_instru = 9.5
      ampm = "AM"
    else:
      lt_instru = 23.5
      ampm = "PM"

  fileversion = "05"
  pl_vars = ["ta", "q"]
  sf_vars = ["sp", "skt"]

  outstr = {
    "temp"  : "L2_temperature_daily_average",
    "h2o"   : "L2_H2O_daily_average",
    "press" : "L2_P_surf_daily_average",
    "stat"  : "L2_status",
  }


  # ... Files and directories ...
  # -----------------------------

  project_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
  )
  dirin_bdd = os.path.normpath(
    os.path.join(project_dir, "input")
    # "/home_local/slipsl/GEWEX/input"
    # "/home_local/slipsl/GEWEX/input"
    # "/bdd/ERA5/NETCDF/GLOBAL_025/hourly"
  )
  dirin_pl = os.path.join(dirin_bdd, "AN_PL")
  dirin_sf = os.path.join(dirin_bdd, "AN_SF")
  dirout   = os.path.normpath(
      "/data/slipsl/GEWEX/"
  )

  if args.verbose:
    print("dirin_pl: ", dirin_pl)
    print("dirin_sf: ", dirin_sf)
    print("dirout  : ", dirout)


  # .. Main program ..
  # ==================

  univT = [i-24+0.5 for i in range(72)]

  date_curr = args.date_start

  while date_curr <= args.date_end:
    date_prev = date_curr - dt.timedelta(days=1)
    date_next = date_curr + dt.timedelta(days=1)

    print(date_curr)
    print(date_prev, date_next)

    pp.pprint(
      [
        get_filein(v, date_curr) for v in pl_vars + sf_vars
      ]
    )

    for var in outstr.values():
      print(get_fileout(var, date_curr))

    read_netcdf(get_filein(sf_vars[0], date_curr))

    date_curr = date_curr + dt.timedelta(days=1)

  print("\n"+80*"=")
  print(f"Run ended in {dt.datetime.now() - run_deb}")

  exit()

