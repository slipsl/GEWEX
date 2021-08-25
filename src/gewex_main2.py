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
# import pandas as pd
# from fortio import FortranFile
from scipy.io import FortranFile
from scipy.interpolate import interp1d
# from netCDF4 import Dataset

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

    return weight, time_indices, (date_bounds.min(), date_bounds.max())


#----------------------------------------------------------------------
def get_wght_mean(V, i, w, t):

  t_min, t_max = t
  w_min, w_max = w

  # print(
  #   t_min, t_max,
  #   w_min, w_max,
  #   V.ncdata[t_min, ..., i].shape,
  #   V.ncdata[t_max, ..., i].shape,
  # )

  return (
    w_min * V.ncdata[t_min, ..., i] +
    w_max * V.ncdata[t_min, ..., i]
  )

  # print(
  #   profile.shape
  # )

  # print(72*".")
  # for j in range(0, profile.size, 20):
  #   print(
  #     j,
  #     w_min,
  #     V.ncdata[t_min, ..., j, i],
  #     w_max,
  #     V.ncdata[t_max, ..., j, i],
  #     profile[..., j]
  #   )


#----------------------------------------------------------------------
def get_interp(V, i, j, ncgrid, tggrid, P0, V0):
  pass

  X = ncgrid.lev
  Y = V.ncprofiles[..., j, i]

  cond = tggrid.lev <= P0

  # tgprofile = np.full((tggrid.nlev, ), np.nan)

  fn = interp1d(
    x=X,
    y=Y,
    fill_value="extrapolate",
  )

  V.tgprofiles[cond, j, i] = fn(tggrid.lev[cond])

  if not V0:
    z = np.squeeze(np.argwhere(cond)[-1])
    V0 = V.tgprofiles[z, j, i]
    # V0 = tgprofile[np.squeeze(np.argwhere(cond)[-1])]


  # print(
  #   F"{tgprofile}\n"
  #   F"{np.argwhere(~np.isnan(tgprofile))}\n"
  #   F"{np.argwhere(cond)}\n"
  #   # F""
  #   F"V0 = {V0}"
  # )

  V.tgprofiles[~cond, j, i] = V0
  # tgprofile[~cond] = V0

  # exit()

  # if variable.name == "temp":
  #   V0 = Tsurf.outvalues[i_lon, i_lat]
  # else:
  #   V0 = tgprofile[np.squeeze(np.argwhere(tgprofile)[-1])]
  # tgprofile[~cond] = V0
  # variable.outvalues[i_lon, i_lat, :tg_grid.nlev] = tgprofile
  # if variable.name == "temp":
  #   variable.outvalues[i_lon, i_lat, tg_grid.nlev+1] = V0

  return
  # return tgprofile


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

  # fg_press = True
  # fg_temp  = True
  # fg_h2o   = True
  fg_temp  = not args.notemp
  fg_h2o   = not args.noh2o

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

    var_list = tuple(
      V for V in (Psurf, Tsurf, T, Q) if V
    )

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

    # ... Compute f(lon, date) stuff ...
    # ----------------------------------
    weight, time_indices, (date_min, date_max) = \
      get_weight_indices(ncgrid.lon, date_curr, instru.tnode)

    freemem()

    code_start = dt.datetime.now()
    for V in var_list:
      print(80*"~")
      print(F"Load nc data for {V.name}")
      V.ncdata = gnc.load_netcdf(
        V, date_min, date_max, params
        # V, date_bounds.min(), date_bounds.max(), params
      )
      freemem()
    code_stop = dt.datetime.now()
    print(code_stop - code_start)


    print(80*"~")
    print(F"Init datas")
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
      for V in var_list:
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
            fg_print = not (i % 60) and not (j % 60)
            if fg_print:
              print(
                F"lon = {ncgrid.lon[i]} ; "
                F"lat = {ncgrid.lat[j]}"
              )

            if V.name == "temp":
              V0 = Tsurf.tgprofiles[j, i]
            else:
              V0 = None

            # V.tgprofiles[..., j, i] = get_interp(
            #   V, i, j,
            #   ncgrid, tggrid,
            #   Psurf.tgprofiles[j, i], V0
            # )
            V.get_interp(
              i, j, ncgrid, tggrid,
              Psurf.tgprofiles[j, i], V0
            )
            # if fg_print:
            #   print(V.name, V.tgprofiles[..., j, i])

    print("Write files")
    for V in var_list:
      fileout = V.pathout(params.dirout, date_curr)
      if fileout:
        print(V.name, fileout)
        values = gw.grid_nc2tg(V.tgprofiles, ncgrid, tggrid)
        # print(values.shape, values.ndim)
        # values = np.rollaxis(values, -1, -2)
        # print(values.shape)
        with FortranFile(
          fileout,
          # V.pathout(params.dirout, date_curr),
          mode="w", header_dtype=">u4"
        ) as f:
          f.write_record(
            np.rollaxis(values, -1, -2).astype(dtype=">f4")
          )


  print("\n"+72*"=")
  print(f"Run ended in {dt.datetime.now() - run_deb}")

  exit()
