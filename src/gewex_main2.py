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
import psutil
import os
from pathlib import Path
import datetime as dt
# from cftime import num2date, date2num
import cftime as cf  # num2date, date2num
import pprint


import numpy as np
import pandas as pd
# from fortio import FortranFile
from scipy.io import FortranFile
from scipy.interpolate import interp1d
from netCDF4 import Dataset

pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================
import gewex_param as gw
import gewex_netcdf as gnc


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
    help="Verbose mode"
  )

  parser.add_argument(
    "-f", "--force", action="store_true",
    help="If output files exsist, replace them."
  )

  parser.add_argument(
    "-p", "--poids", action="store_true",
    help="Produce weight file."
  )

  # parser.add_argument("-d", "--dryrun", action="store_true",
  #                     help="only print what is to be done")
  return parser.parse_args()


#----------------------------------------------------------------------
def freemem():

  # # gives a single float value
  # print(psutil.cpu_percent())
  # # gives an object with many fields
  # print(psutil.virtual_memory())
  # # you can convert that object to a dictionary 
  # print(dict(psutil.virtual_memory()._asdict()))
  # # you can have the percentage of used RAM
  # print(psutil.virtual_memory().percent)
  # # you can calculate percentage of available memory
  # print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

  # you can calculate percentage of available memory
  free = (
    psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
  )
  used = psutil.virtual_memory().used / 1024**2
  print(
    F"Memory: Free = {free:.2f} % ; "
    F"Used = {used:.2f} Mb"
  )
  # print(
  #   F"Free memory = "
  #   F"{psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:6.2f}"
  #   F"%"
  # )


#----------------------------------------------------------------------
def lon2tutc(lon, date, node):

  if lon > 180.:
    lon = lon - 360.

  offset = 0.5
  hours = (node - offset - lon / 15.)

  return date + dt.timedelta(hours=hours)


#----------------------------------------------------------------------
def utc2min(date):
  return date.replace(minute=0, second=0, microsecond=0)


#----------------------------------------------------------------------
def utc2max(date):

  return (
    # date.replace(minute=0, second=0, microsecond=0) +
    utc2min(date) + dt.timedelta(hours=1)
  )


#----------------------------------------------------------------------
def date2weight(date, date1, date2):

  (a, b) = (date - date1, date2 - date)

  return (
      (1. - (a / (a + b))),
      (1. - (b / (a + b))),
    )


#----------------------------------------------------------------------
def date2idx(date, date1, date2):

  return [
      int((d - date).total_seconds() / 3600.)
      for d in (date1, date2)
  ]


#----------------------------------------------------------------------
def dt_bounds(date, mode="full"):

  if mode not in ["date", "weight", "full"]:
    raise("Invalid argument")

  date_min = date.replace(minute=0, second=0, microsecond=0)
  date_max = date_min + dt.timedelta(hours=1)

  delta_min = (date - date_min).total_seconds()
  delta_max = (date_max - date).total_seconds()

  weight_min = 1. - (delta_min / (delta_min + delta_max))
  weight_max = 1. - (delta_max / (delta_min + delta_max))

  if mode == "date":
    ret = (date_min, date_max, )
  elif mode == "weight":
    ret = (weight_min, weight_max, )
  else:
    ret = (
      (date_min, weight_min),
      (date_max, weight_max)
    )

  return ret


#----------------------------------------------------------------------
def dt_idx(dates):

  step = 24  # netcdf number of time steps per day

  return (d.hour + 24 * (d.day - 1) for d in dates)


#----------------------------------------------------------------------
def dt_weight(date):

  (date_min, date_max) = dt_bounds(date)
  delta_min = (date - date_min).total_seconds()
  delta_max = (date_max - date).total_seconds()

  weight_min = 1. - (delta_min / (delta_min + delta_max))
  weight_max = 1. - (delta_max / (delta_min + delta_max))

  return (weight_min, weight_max)


#----------------------------------------------------------------------
def iter_dates(start, stop):

  delta = 1 + (stop - start).days
  return (start + dt.timedelta(days=i) for i in range(delta))


#----------------------------------------------------------------------
def num2date(val):

  # return dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(val))
  return cf.num2date(
    val,
    units=ncgrid.tunits,
    calendar=ncgrid.calendar,  # 'standard',
    only_use_cftime_datetimes=False,  #True,
    only_use_python_datetimes=True,  # False,
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

  if args.runtype == 5:
    args.force = True

  # ... Constants ...
  # -----------------

  # ... Files and directories ...
  # -----------------------------
  project_dir = Path(__file__).resolve().parents[1]
  # dirin = project_dir.joinpath("input")
  # # dirin_3d = dirin.joinpath("AN_PL")
  # # dirin_2d = dirin.joinpath("AN_SF")
  # dirout   = project_dir.joinpath("output")

  instru = gw.InstruParam(args.runtype)
  params = gw.GewexParam(project_dir)

  if args.verbose:
    print(instru)
    print(params)

  fg_press = True
  fg_temp  = True
  fg_h2o   = True


  # .. Main program ..
  # ==================

  # ... Initialize things ...
  # -------------------------

  ncgrid = gnc.NCGrid()
  tggrid = gw.TGGrid()

  for date_curr in iter_dates(args.date_start, args.date_end):
    date_prev = date_curr - dt.timedelta(days=1)
    date_next = date_curr + dt.timedelta(days=1)

    if args.verbose:
      print(
        F"{72*'='}\n{date_prev} < {date_curr} > {date_next}\n{72*'-'}"
      )

    fg_process = True

    ncfiles = []
    outfiles = []

    dates = (date_prev, date_curr, date_next)

    Psurf = gw.Variable("Psurf", instru)
    ncfiles.append(Psurf.get_ncfiles(params.dirin, dates))
    outfiles.append(Psurf.pathout(params.dirout, date_curr))

    if fg_temp:
      Tsurf = gw.Variable("Tsurf", instru)
      ncfiles.append(Tsurf.get_ncfiles(params.dirin, dates))
      T = gw.Variable("temp", instru)
      ncfiles.append(T.get_ncfiles(params.dirin, dates))
      outfiles.append(T.pathout(params.dirout, date_curr))
    else:
      Tsurf = None
      T = None

      stat = gw.Variable("stat", instru)
      outfiles.append(stat.pathout(params.dirout, date_curr))

    if fg_h2o:
      Q = gw.Variable("h2o", instru)
      ncfiles.append(Q.get_ncfiles(params.dirin, dates))
      outfiles.append(Q.pathout(params.dirout, date_curr))
    else:
      Q = None

    # ... Check input files ...
    # -------------------------
    ncfiles = np.unique(ncfiles)
    filesok = [f.exists() for f in ncfiles]
    missfiles = np.ma.array(
      ncfiles,
      # mask=np.logical_not(filesok)
      mask=filesok
    )
    if not all(filesok):
      print(F"Missing input file(s), skip date")
      for file in missfiles[~missfiles.mask]:
        print(F"  - {file}")
      continue

    # ... Check output files ...
    # --------------------------
    filesok = [f.exists() for f in outfiles]
    donefiles = np.ma.array(
      outfiles,
      mask=np.logical_not(filesok)
      # mask=filesok
    )
    if any(filesok):
      # print(F"Onput file(s) already there, skip date")
      print(F"Onput file(s) already there", end="")
      if args.force:
        print(F", they will be replaced.")
      else:
        print(F", skip date.")
      if args.verbose:
        for file in donefiles[~donefiles.mask]:
          print(F"  - {file}")

      if not args.force:
        continue

    # ... Output directory ...
    # ------------------------
    subdir = Psurf.dirout(params.dirout, date_curr)
    if not subdir.exists():
      if args.verbose:
        print(F"Create output subdirectory: {subdir}")
      subdir.mkdir(parents=True, exist_ok=True)

    # ... Load NetCDF & target grids ...
    # ----------------------------------
    if not ncgrid.loaded:
      if T:
        variable = T
      elif Q:
        variable = Q
      else:
        variable = Psurf
      ncgrid.load(variable.get_ncfiles(params.dirin, args.date_start))

    if not tggrid.loaded:
      tggrid.load(ncgrid)

    date_utc = np.array([
      lon2tutc(l, date_curr, instru.tnode)
      for l in ncgrid.lon
    ])

    date_bounds = np.array([
      (utc2min(d), utc2max(d))
      for d in date_utc
    ])

    weight = np.array([
      date2weight(d, d1, d2)
      for d, (d1, d2) in zip(date_utc, date_bounds)
    ])

    time_indices = np.array([
      date2idx(date_bounds.min(), d1, d2)
      for (d1, d2) in date_bounds
    ])

    var_list = tuple(
      V for V in (Psurf, Tsurf, T, Q) if V
    )

    freemem()

    code_start = dt.datetime.now()
    for V in var_list:
      print(80*"~")
      print(F"Load nc data for {V.name}")
      V.ncdata = gnc.load_netcdf(
        V, date_bounds.min(), date_bounds.max(), params
      )
      freemem()
    code_stop = dt.datetime.now()
    print(code_stop - code_start)


    for V in var_list:
      print(V.ncdata.shape)
      V.init_datas(ncgrid, tggrid)
    freemem()


    for i in range(ncgrid.nlon):
      fg_print = not (i % 60)

      if fg_print:
        print(
          F"lon = {ncgrid.lon[i]}"
          # F"lat = {nc_grid.lat[i_lat]}"
        )

      if fg_print:
        print("Weighted nc mean")

      if fg_print:
        print("for lat, interp")

  print("\n"+72*"=")
  print(f"Run ended in {dt.datetime.now() - run_deb}")

  exit()
