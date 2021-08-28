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
import cftime as cf  # num2date, date2num
import pprint

import numpy as np
# import pandas as pd
# from fortio import FortranFile
from scipy.io import FortranFile
# from scipy.interpolate import interp1d
# from netCDF4 import Dataset

pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================
import gewex_param as gwp
import gewex_variable as gwv
import gewex_netcdf as gwn


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
  used = psutil.virtual_memory().used
  used2 = psutil.virtual_memory().total - psutil.virtual_memory().available
  if used > 1024**3:
    coeff = 1024**3
    units = "gb"
  else:
    coeff = 1024**2
    units = "mb"
  used = used / coeff
  used2 = used2 / coeff
  print(
    F"Memory: Free = {free:.2f} % ; "
    F"Used = {used:.2f} {units}"
    F" / Used = {used:.2f} {units}"
  )
  # print(
  #   F"Free memory = "
  #   F"{psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:6.2f}"
  #   F"%"
  # )


#----------------------------------------------------------------------
def date_prev(date):

  return date - dt.timedelta(days=1)


#----------------------------------------------------------------------
def date_next(date):

  return date + dt.timedelta(days=1)


#----------------------------------------------------------------------
def check_inputs(V_list, date, dirin):

  f_list = []
  for V in V_list:
    f_list.extend([
      f for f in V.get_ncfiles(
        dirin,
        (date_prev(date), date, date_next(date))
      )
      if not f.exists()
    ])

  return f_list


#----------------------------------------------------------------------
def check_outputs(V_list, date, dirout):

  f_list = []
  for V in V_list:
    if (V.pathout(dirout, date) and 
        V.pathout(dirout, date).exists()):
      f_list.append(V.pathout(dirout, date))

  return f_list


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
def get_weight_indices(lon, date, node):

    date_utc = np.array([
      lon2tutc(l, date, node)
      for l in lon
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

    # for l, d0, (d1, d2), (w1, w2), (t1, t2) in zip(
    #   lon, date_utc, date_bounds, weight, time_indices
    # ):
    #   print(
    #     F"{l:.2f} °E / {node} h = "
    #     F"{w1:.2f} x {d1}"
    #     F" < {d0} > "
    #     F"{w2:.2f} x {d2}"
    #     F"   ({t1} {t2})"
    #   )
    # exit()

    return weight, time_indices, (date_bounds.min(), date_bounds.max())


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

  # if args.runtype == 5:
  #   args.force = True

  # ... Constants ...
  # -----------------

  # ... Files and directories ...
  # -----------------------------
  project_dir = Path(__file__).resolve().parents[1]
  # dirin = project_dir.joinpath("input")
  # # dirin_3d = dirin.joinpath("AN_PL")
  # # dirin_2d = dirin.joinpath("AN_SF")
  # dirout   = project_dir.joinpath("output")

  instru = gwp.InstruParam(args.runtype)
  params = gwp.GewexParam(project_dir)

  if args.verbose:
    print(instru)
    print(params)

  # fg_press = True
  # fg_temp  = True
  # fg_h2o   = True
  fg_temp  = not args.notemp
  fg_h2o   = not args.noh2o

  # .. Main program ..
  # ==================

  # ... Initialize things ...
  # -------------------------
  # Grids
  ncgrid = gwn.NCGrid()
  tggrid = gwv.TGGrid()
  # Variables
  Psurf = gwv.Variable("Psurf", instru)
  stat = gwv.Variable("stat", instru)
  time = gwv.Variable("time", instru)
  if fg_temp:
    Tsurf = gwv.Variable("Tsurf", instru)
    T = gwv.Variable("temp", instru)
  else:
    Tsurf = T = None
  if fg_h2o:
    Q = gwv.Variable("h2o", instru)
  else:
    Q = None

  V_list = tuple(
    V for V in (Psurf, Tsurf, T, Q) if V
  )


  # ... Process date ...
  # --------------------
  for date_curr in iter_dates(args.date_start, args.date_end):

    date_deb = dt.datetime.now()

    # if args.verbose:
    print(
      F"{72*'='}\n"
      F"{date_prev(date_curr):%Y-%m-%d}"
      F" < {date_curr:%Y-%m-%d} > "
      F"{date_next(date_curr):%Y-%m-%d}"
      F"\n{72*'-'}"
    )

    # print(72*"=")
    # print(72*"=")

    # pp.pprint(locals())

    # print(72*"=")
    # print(72*"=")

    # ... Check output files ...
    # --------------------------
    f_list = check_outputs(V_list, date_curr, params.dirout)
    if f_list:
      print(F"Onput file(s) already there", end="")
      if args.force:
        print(F", they will be replaced.")
      else:
        print(F", skip date.")
      if args.verbose:
        for f in f_list:
          print(F"  - {f}")
      if not args.force:
        continue

    # ... Check input files ...
    # -------------------------
    f_list = check_inputs(V_list, date_curr, params.dirin)
    if f_list:
      print(F"Missing input file(s), skip date")
      for f in set(f_list):
        print(F"  - {f}")
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
        V = T
      elif Q:
        V = Q
      else:
        V = Psurf
      if args.verbose:
        print(
          F"Load grid from "
          F"{V.get_ncfiles(params.dirin, args.date_start)}\n"
          F"{72*'='}"
        )
      ncgrid.load(V.get_ncfiles(params.dirin, args.date_start))

    if not tggrid.loaded:
      tggrid.load()

    # ... Compute f(lon, date) stuff ...
    # ----------------------------------
    weight, time_indices, (date_min, date_max) = \
      get_weight_indices(ncgrid.lon, date_curr, instru.tnode)

    # ... Init arrays for variables data ...
    # --------------------------------------
    if args.verbose:
      print(F"{72*'~'}\nInit datas")
    for V in V_list + (stat, ):
      V.init_datas(ncgrid, tggrid)
    freemem()

    # ... Load netcdf data ...
    # ------------------------
    freemem()
    if args.verbose:
      code_start = dt.datetime.now()
    for V in (time, ) + V_list:
      if args.verbose:
        print(F"{72*'~'}\nLoad nc data for {V.name}")
      V.ncdata = gwn.load_netcdf(
        V, date_min, date_max, params
      )
      if args.verbose:
        print(V.ncdata.shape)
      freemem()
    if args.verbose:
      code_stop = dt.datetime.now()
      print(code_stop - code_start)

    # ... Loop over netcdf longitudes ...
    # -----------------------------------
    for i in range(ncgrid.nlon):
      fg_print = not (i % 60) and args.verbose

      if fg_print:
        print(F"lon = {ncgrid.lon[i]}")
        print(
          num2date(time.ncdata[time_indices[i][0]]),
          weight[i][0],
          num2date(time.ncdata[time_indices[i][1]]),
          weight[i][1],
        )

      if fg_print:
        print("Weighted nc mean")
      for V in V_list:
        V.get_wght_mean(i, weight[i], time_indices[i])

        if V.mode == "2d":
          V.tgprofiles[..., i] = V.ncprofiles[..., i]
          if fg_print:
            print(
              V.name,
              np.nanmin(V.tgprofiles),
              np.nanmax(V.tgprofiles),
              np.nanmean(V.tgprofiles),
            )
        else:
          if fg_print:
            print("for lat, interp")
          for j in range(ncgrid.nlat):
            fg_print = not (i % 60) and not (j % 60) and args.verbose
            if fg_print:
              print(
                F"lon = {ncgrid.lon[i]} ; "
                F"lat = {ncgrid.lat[j]}"
              )

            if V.name == "temp":
              V0 = Tsurf.tgprofiles[j, i]
            else:
              V0 = None

            V.get_interp(
              i, j, ncgrid, tggrid,
              Psurf.tgprofiles[j, i], V0
            )
            # if fg_print:
            #   print(V.name, V.tgprofiles[..., j, i])

    stat.tgprofiles[...] = 10000

    if args.verbose:
      print("Write files")
    for V in V_list + (stat, ):
      fileout = V.pathout(params.dirout, date_curr)
      if fileout:
        if args.verbose:
          print(V.name, fileout)
        values = gwv.grid_nc2tg(V.tgprofiles, ncgrid, tggrid)
        with FortranFile(fileout, mode="w", header_dtype=">u4") as f:
          f.write_record(
            np.rollaxis(values, -1, -2).astype(dtype=">f4")
          )

    # ... Some cleaning to free memory ...
    # ------------------------------------
    if args.verbose:
      print(F"{72*'~'}\nClear datas")
    for V in V_list + (stat, ):
      V.clear_datas()
    freemem()

    print(
      F"{72*'-'}\n"
      F"{date_curr:%Y-%m-%d} processed in "
      F"{dt.datetime.now() - date_deb}"
    )

  print(F"\n{72*'='}\nRun ended in {dt.datetime.now() - run_deb}")

  exit()
