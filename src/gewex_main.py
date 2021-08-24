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
    "-p", "--poids", action="store_true",
    help="Produce weight file."
  )

  # parser.add_argument("-d", "--dryrun", action="store_true",
  #                     help="only print what is to be done")
  return parser.parse_args()


#----------------------------------------------------------------------
def print_pl(var1, var2):

  # print(F"{nc_lev[0]:5.2f}  {var1[0]:11.4e}")
  # print(F"{nc_lev[1]:5.2f}  {var1[1]:11.4e}")
  # print(F"{nc_lev[2]:5.2f}  {var1[2]:11.4e}")
  # print(F"{nc_lev[3]:5.2f}  {var1[3]:11.4e}")
  # print(F"{nc_lev[4]:5.2f}  {var1[4]:11.4e}")
  # print(F"{nc_lev[5]:5.2f}  {var1[5]:11.4e}")
  # print(F"{nc_lev[6]:5.2f}  {var1[6]:11.4e}")
  # print(F"{nc_lev[7]:5.2f}  {var1[7]:11.4e}")
  print(F"{nc_lev[8]:6.2f}  {var1[8]:11.4e}")
  print(F"{18*' '}  {var2[0]:11.4e}  {P_tigr[0]:6.2f}")
  print(F"{nc_lev[9]:6.2f}  {var1[9]:11.4e}")
  print(F"{18*' '}  {var2[1]:11.4e}  {P_tigr[1]:6.2f}")
  print(F"{nc_lev[10]:6.2f}  {var1[10]:11.4e}")
  print(F"{18*' '}  {var2[2]:11.4e}  {P_tigr[2]:6.2f}")
  print(F"{nc_lev[11]:6.2f}  {var1[11]:11.4e}")
  print(F"{18*' '}  {var2[3]:11.4e}  {P_tigr[3]:6.2f}")
  print(F"{nc_lev[12]:6.2f}  {var1[12]:11.4e}")
  print(F"{18*' '}  {var2[4]:11.4e}  {P_tigr[4]:6.2f}")
  print(F"{nc_lev[13]:6.2f}  {var1[13]:11.4e}")
  print(
    F"{nc_lev[14]:6.2f}  {var1[14]:11.4e}   "
    F"{var2[5]:11.4e}  {P_tigr[5]:6.2f}"
  )
  print(F"{18*' '}  {var2[6]:11.4e}  {P_tigr[6]:6.2f}")
  print(F"{nc_lev[15]:6.2f}  {var1[15]:11.4e}")
  print(F"{18*' '}  {var2[7]:11.4e}  {P_tigr[7]:6.2f}")
  print(F"{nc_lev[16]:6.2f}  {var1[16]:11.4e}")
  print(F"{18*' '}  {var2[8]:11.4e}  {P_tigr[8]:6.2f}")
  print(F"{nc_lev[17]:6.2f}  {var1[17]:11.4e}")
  print(F"{18*' '}  {var2[9]:11.4e}  {P_tigr[9]:6.2f}")
  print(F"{18*' '}  {var2[10]:11.4e}  {P_tigr[10]:6.2f}")
  print(F"{nc_lev[18]:6.2f}  {var1[18]:11.4e}")
  print(F"{18*' '}  {var2[11]:11.4e}  {P_tigr[11]:6.2f}")
  print(F"{nc_lev[19]:6.2f}  {var1[19]:11.4e}")
  print(F"{18*' '}  {var2[12]:11.4e}  {P_tigr[12]:6.2f}")
  print(F"{nc_lev[20]:6.2f}  {var1[20]:11.4e}")
  print(F"{18*' '}  {var2[13]:11.4e}  {P_tigr[13]:6.2f}")
  print(F"{nc_lev[21]:6.2f}  {var1[21]:11.4e}")
  print(F"{18*' '}  {var2[14]:11.4e}  {P_tigr[14]:6.2f}")
  print(F"{nc_lev[22]:6.2f}  {var1[22]:11.4e}")
  print(F"{18*' '}  {var2[15]:11.4e}  {P_tigr[15]:6.2f}")
  print(F"{nc_lev[23]:6.2f}  {var1[23]:11.4e}")
  print(F"{nc_lev[24]:6.2f}  {var1[24]:11.4e}")
  print(F"{18*' '}  {var2[16]:11.4e}  {P_tigr[16]:6.2f}")
  print(F"{nc_lev[25]:6.2f}  {var1[25]:11.4e}")
  print(F"{18*' '}  {var2[17]:11.4e}  {P_tigr[17]:6.2f}")
  print(F"{nc_lev[26]:6.2f}  {var1[26]:11.4e}")
  print(F"{nc_lev[27]:6.2f}  {var1[27]:11.4e}")
  print(
    F"{nc_lev[28]:6.2f}  {var1[28]:11.4e}   "
    F"{var2[18]:11.4e}  {P_tigr[18]:6.2f}"
    )
  print(F"{nc_lev[29]:6.2f}  {var1[29]:11.4e}")
  print(F"{18*' '}  {var2[19]:11.4e}  {P_tigr[19]:6.2f}")
  print(F"{nc_lev[30]:6.2f}  {var1[30]:11.4e}")
  print(F"{nc_lev[31]:6.2f}  {var1[31]:11.4e}")
  print(F"{nc_lev[32]:6.2f}  {var1[32]:11.4e}")
  print(F"{18*' '}  {var2[20]:11.4e}  {P_tigr[20]:6.2f}")
  print(F"{nc_lev[33]:6.2f}  {var1[33]:11.4e}")
  print(F"{nc_lev[34]:6.2f}  {var1[34]:11.4e}")
  print(F"{18*' '}  {var2[21]:11.4e}  {P_tigr[21]:6.2f}")
  print(F"{nc_lev[35]:6.2f}  {var1[35]:11.4e}")
  print(F"{nc_lev[36]:6.2f}  {var1[36]:11.4e}")
  print(F"{18*' '}  {var2[22]:11.4e}  {P_tigr[22]:6.2f}")


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
  print(
    F"Free memory = "
    F"{psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:6.2f}"
    F"%"
  )


# #----------------------------------------------------------------------
# def date2num(val):

#   return dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(val))


#----------------------------------------------------------------------
def lon2tutc(lon, date, tnode):

  # # Convert local time in decimal hours to hh:mm:ss
  # (m, s) = divmod(tnode * 3600, 60)
  # (h, m) = divmod(m, 60)

  # if lon >= 180.:
  if lon > 180.:
    lon = lon - 360.
  t_local = date + dt.timedelta(hours=tnode)

  offset = 0.5

  return t_local  - dt.timedelta(hours=(offset + lon/15.))


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

  return (d.hour + step * (d.day - 1) for d in dates)


#----------------------------------------------------------------------
def dt_weight(date):

  (date_min, date_max) = dt_bounds(date)
  delta_min = (date - date_min).total_seconds()
  delta_max = (date_max - date).total_seconds()

  weight_min = 1. - (delta_min / (delta_min + delta_max))
  weight_max = 1. - (delta_max / (delta_min + delta_max))

  return (weight_min, weight_max)


#----------------------------------------------------------------------
def compute_outdata(variable, w_min, w_max, grid, fg_print):

  if variable.mode == "2d":
    shape = (grid.nlat, )
  else:
    shape = (grid.nlat, grid.nlev, )
  outvalues = np.empty(shape)

  if fg_print:
    print(F"inside compute: {variable.ncvalues.shape} => {outvalues.shape}")

  outvalues = (
    w_min * variable.ncvalues[..., 0] +
    w_max * variable.ncvalues[..., 1]
  )
  if fg_print:
    print("sortie de compute:", type(outvalues), outvalues.shape)

  return outvalues


#----------------------------------------------------------------------
def iter_dates(start, stop):

  delta = 1 + (stop - start).days
  return (start + dt.timedelta(days=i) for i in range(delta))


#----------------------------------------------------------------------
def ncgrid2tggrid(variable, nc_grid, tg_grid):
  pass


#----------------------------------------------------------------------
def num2date(val):

  # return dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(val))
  return cf.num2date(
    val,
    units=nc_grid.tunits,
    calendar=nc_grid.calendar,  # 'standard',
    only_use_cftime_datetimes=False,  #True,
    only_use_python_datetimes=True,  # False,
    has_year_zero=None
  )


#######################################################################

if __name__ == "__main__":

  run_deb = dt.datetime.now()

  freemem()

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

  print(instru)
  print(params)

  fg_press = True
  fg_temp  = True
  fg_h2o   = True


  # .. Main program ..
  # ==================

  # ... Initialize things ...
  # -------------------------

  nc_grid = gnc.NCGrid()
  tg_grid = gw.TGGrid()

  for date_curr in iter_dates(args.date_start, args.date_end):
    date_prev = date_curr - dt.timedelta(days=1)
    date_next = date_curr + dt.timedelta(days=1)

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

      for file in donefiles[~donefiles.mask]:
        print(F"  - {file}")

      if not args.force:
        continue

    # ... Output directory ...
    # ------------------------
    subdir = Psurf.dirout(params.dirout, date_curr)
    if not subdir.exists():
      print(F"Create output subdirectory: {subdir}")
      subdir.mkdir(parents=True, exist_ok=True)

    # ... Load NetCDF & target grids ...
    # ----------------------------------
    if not nc_grid.loaded:
      if T:
        variable = T
      elif Q:
        variable = Q
      else:
        variable = Psurf
      nc_grid.load(variable.get_ncfiles(params.dirin, args.date_start))

    if not tg_grid.loaded:
      tg_grid.load(nc_grid)

    for i_lon, lon in enumerate(nc_grid.lon):

      fg_print = False
      # if i_lon in range(700, 721):
      if not (i_lon % 60):
        fg_print = True

      if fg_print:
        print(F"\n{72*'~'}")

      # Get dates closest to local time (f(lon)) and associated weight
      # => to perform weighted mean
      dt_utc = lon2tutc(lon, date_curr, instru.tnode)
      if fg_print:
        print(F"UTC = {dt_utc}")
      ((dt_min, w_min), (dt_max, w_max)) = dt_bounds(dt_utc)
      (t_min, t_max) = dt_idx((dt_min, dt_max))

      if fg_print:
        print(
          F"Calcul poids/dates : "
          F"{w_min:4.2f} x {dt_min} (t={t_min})"
          F" < {dt_utc} < "
          F"{w_max:4.2f} x {dt_max} (t={t_max})"
        )

      # Initialize wanted variables
      if dt_min.month == dt_max.month:
        dates = dt_min
      else:
        dates = (dt_min, dt_max)

      for variable in (Psurf, Tsurf, T, Q):
        if fg_print:
          print(F"Find NetCDF files to read")
        if variable:
          # Find which NetCDF file(s) contain wanted date(s)
          variable.ncfiles = variable.get_ncfiles(params.dirin, dates)
          # pp.pprint(variable.ncfiles)
          # initialize output variables
          if variable.outvalues is None:
            variable.init_w_mean(nc_grid)
            variable.init_outval(tg_grid)


      # Compute weighted mean
      for variable in (Psurf, Tsurf, T, Q):
        if variable:
          if fg_print:
            print(F"Read {variable.name}, lon = {lon} ({i_lon})")
          variable.ncvalues = gnc.read_netcdf(
            variable, nc_grid, i_lon, (t_min, t_max)
          )
          if fg_print:
            variable.print_values()
          variable.w_mean[i_lon, ...] = compute_outdata(
            variable, w_min, w_max, nc_grid, fg_print
          )
          if variable.mode == "2d":
            variable.outvalues[i_lon, ...] = variable.w_mean[i_lon, ...]

      # Interpolate profile
      for i_lat in range(nc_grid.nlat):
        if fg_print and not (i_lat % 60):
          print(
            F"lon = {nc_grid.lon[i_lon]} / "
            F"lat = {nc_grid.lat[i_lat]}"
          )
        for variable in (T, Q):
          if variable:
            tgprofile = np.ma.masked_all((tg_grid.nlev, ))
            cond = tg_grid.lev <= Psurf.outvalues[i_lon, i_lat]
            fn = interp1d(
              x=nc_grid.lev,
              y=variable.w_mean[i_lon, i_lat, :],
              fill_value="extrapolate",
            )
            tgprofile[cond] = fn(np.ma.getdata(tg_grid.lev[cond]))
            if variable.name == "temp":
              V0 = Tsurf.outvalues[i_lon, i_lat]
            else:
              V0 = tgprofile[np.squeeze(np.argwhere(tgprofile)[-1])]
            tgprofile[~cond] = V0
            variable.outvalues[i_lon, i_lat, :tg_grid.nlev] = tgprofile
            if variable.name == "temp":
              variable.outvalues[i_lon, i_lat, tg_grid.nlev+1] = V0




        # if fg_print and not (i_lat % 60):
        #   print(T.outvalues[i_lon, i_lat, :])




      # exit()


      # # Psurf
      # # ===============================================================
      # # Read input data
      # # ---------------
      # if fg_print:
      #   print(F"Read {Psurf.name}, lon = {lon} ({i_lon})")
      # Psurf.ncvalues = gnc.read_netcdf(
      #   Psurf, nc_grid, i_lon, (t_min, t_max)
      # )
      # if fg_print:
      #   Psurf.print_values()
      # # Compute output data
      # # -------------------
      # Psurf.outvalues[i_lon, ...] = compute_outdata(
      #   Psurf, w_min, w_max, nc_grid, fg_print
      # )

      # # Tsurf & T
      # # ===============================================================
      # if fg_temp:
      #   if fg_print:
      #     print(F"Read {Tsurf.name}, lon = {lon} ({i_lon})")
      #   Tsurf.ncvalues = gnc.read_netcdf(
      #     Tsurf, nc_grid, i_lon, (t_min, t_max)
      #   )
      #   if fg_print:
      #     Tsurf.print_values()
      #   Tsurf.outvalues[i_lon, ...] = compute_outdata(
      #     Tsurf, w_min, w_max, nc_grid, fg_print
      #   )

      #   if fg_print:
      #     print(F"Read {T.name}, lon = {lon} ({i_lon})")
      #   T.ncvalues = gnc.read_netcdf(
      #     T, nc_grid, i_lon, (t_min, t_max)
      #   )
      #   if fg_print:
      #     T.print_values()
      #   if fg_print:
      #     print(F"Interpolate")
      #   outvalues = compute_outdata(T, w_min, w_max, nc_grid, fg_print)

      #   for i_lat, (ncprofile, P0, T0) in enumerate(zip(
      #     outvalues, Psurf.outvalues[i_lon, ...], Tsurf.outvalues[i_lon, ...]
      #   )):
      #     if fg_print and not (i_lat % 60):
      #       print(f"Lat = {nc_grid.lat[i_lat]} ({i_lat})")

      #     tgprofile = np.ma.masked_all((tg_grid.nlev, ))
      #     cond = tg_grid.lev <= P0
      #     fn = interp1d(
      #       x=nc_grid.lev,
      #       y=ncprofile,
      #       fill_value="extrapolate",
      #     )
      #     tgprofile[cond] = fn(np.ma.getdata(tg_grid.lev[cond]))

      #     # Y0 = tgprofile[np.squeeze(np.argwhere(tgprofile)[-1])]
      #     # tgprofile[~cond] = Y0
      #     tgprofile[~cond] = T0
      #     # if fg_print and not (i_lat % 60):
      #     #   print(tgprofile)

      #     T.outvalues[i_lon, i_lat, :tg_grid.nlev] = tgprofile
      #     T.outvalues[i_lon, i_lat, tg_grid.nlev+1] = T0
      #     # if fg_print and not (i_lat % 60):
      #     #   print(T.outvalues[i_lon, i_lat, :])

      # # Q
      # # ===============================================================
      # if fg_h2o:
      #   if fg_print:
      #     print(F"Read {Q.name}, lon = {lon} ({i_lon})")
      #   Q.ncvalues = gnc.read_netcdf(
      #     Q, nc_grid, i_lon, (t_min, t_max)
      #   )
      #   if fg_print:
      #     Q.print_values()
      #   if fg_print:
      #     print(F"Interpolate")
      #   outvalues = compute_outdata(Q, w_min, w_max, nc_grid, fg_print)

      #   for i_lat, (ncprofile, P0) in enumerate(zip(
      #     outvalues, Psurf.outvalues[i_lon, ...]
      #   )):
      #     if fg_print and not (i_lat % 60):
      #       print(f"Lat = {nc_grid.lat[i_lat]} ({i_lat})")

      #     tgprofile = np.ma.masked_all((tg_grid.nlev, ))
      #     cond = tg_grid.lev <= P0
      #     fn = interp1d(
      #       x=nc_grid.lev,
      #       y=ncprofile,
      #       fill_value="extrapolate",
      #     )
      #     tgprofile[cond] = fn(np.ma.getdata(tg_grid.lev[cond]))

      #     Q0 = tgprofile[np.squeeze(np.argwhere(tgprofile)[-1])]
      #     tgprofile[~cond] = Q0
      #     # if fg_print and not (i_lat % 60):
      #     #   print(tgprofile)

      #     Q.outvalues[i_lon, i_lat, :] = tgprofile


    print(Psurf.outvalues[:, 180])

    # values = np.roll(Psurf.outvalues, -721, axis=-1)
    # values = np.flip(values, axis=-2)
    values = gw.grid_nc2tg(Psurf.outvalues, nc_grid, tg_grid)
    print(
      np.rollaxis(values, -1, 0).astype(dtype=">f4").shape
    )
    # f_out = FortranFile(fileout, mode="w")
    # with FortranFile("Ptest.dat", mode="w", header_dtype=">u4") as f:
    with FortranFile(
      Psurf.pathout(params.dirout, date_curr),
      mode="w", header_dtype=">u4"
    ) as f:
      f.write_record(values.astype(dtype=">f4"))
      # f.write_record(values.T.astype(dtype=">f4"))

    if fg_temp:
      values = gw.grid_nc2tg(T.outvalues, nc_grid, tg_grid)
      print(
        np.rollaxis(values, -1, 0).astype(dtype=">f4").shape
      )
      with FortranFile(
        T.pathout(params.dirout, date_curr),
        mode="w", header_dtype=">u4"
      ) as f:
        f.write_record(
          np.rollaxis(values, -1, 0).astype(dtype=">f4")
        )

    if fg_h2o:
      values = gw.grid_nc2tg(Q.outvalues, nc_grid, tg_grid)
      print(
        np.rollaxis(values, -1, 0).astype(dtype=">f4").shape
      )
      with FortranFile(
        Q.pathout(params.dirout, date_curr),
        mode="w", header_dtype=">u4"
      ) as f:
        f.write_record(
          np.rollaxis(values, -1, 0).astype(dtype=">f4")
          # np.rollaxis(values, 2, 1).astype(dtype=">f4")
        )




    if args.poids:
      print("Poids", date_curr)
      # To get -180. < lon < +180.
      lon = nc_grid.lon
      lon = np.empty([nc_grid.nlon, ], dtype=float)
      cond = nc_grid.lon[:] > 180.
      lon[cond] = nc_grid.lon[cond] - 360.
      lon[~cond] = nc_grid.lon[~cond]

      idx_lon_l = np.empty([nc_grid.nlon, ], dtype=int)
      idx_lon_r = np.empty([nc_grid.nlon, ], dtype=int)
      weight_l  = np.empty([nc_grid.nlon, ], dtype=float)
      weight_r  = np.empty([nc_grid.nlon, ], dtype=float)

      # Three days worth of timesteps (ts = 1 hour)
      # univT = [i - 24. + 0.5 for i in range(72)]

      with open(F"poids_{instru.name}_{instru.ampm}.dat", "w") as f:
        for idx, lon in enumerate(lon):
          fg_print = False
          if not (idx % 60):
            fg_print = True

          if fg_print:
            print(idx, lon)
          # print(instru.tnode)

          # localT = univT + lon / 15.
          # deltaT = abs(localT - lt_instru)
          deltaT = [
            abs((i - 24.+ 0.5 + lon/15.) - instru.tnode)
            for i in range(72)
          ]

          (imin1, imin2) = np.argsort(deltaT)[0:2]

          w1 = 1. - (deltaT[imin1] / (deltaT[imin1] + deltaT[imin2]))
          w2 = 1. - (deltaT[imin2] / (deltaT[imin1] + deltaT[imin2]))

          idx_lon_l[idx] = imin1
          idx_lon_r[idx] = imin2
          weight_l[idx]  = w1
          weight_r[idx]  = w2

          offset = (date_prev.day - 1) * 24
          count = 3 * 24
          times = nc_grid.time[offset:offset+count]
          date1 = num2date(times[imin1])
          date2 = num2date(times[imin2])

          dt_utc = lon2tutc(lon, date_curr, instru.tnode)
          t_min = dt_utc.replace(minute=0, second=0, microsecond=0)
          t_max = t_min + dt.timedelta(hours=1)

          delta_min = (dt_utc - t_min).total_seconds()
          delta_max = (t_max - dt_utc).total_seconds()
          w_min = 1. - (delta_min / (delta_min + delta_max))
          w_max = 1. - (delta_max / (delta_min + delta_max))

          (t_min, t_max) = (
            d.hour + 24 * (d.day - 1) for d in (t_min, t_max)
          )

          times = nc_grid.time[t_min:t_max+1]
          date_min = num2date(times[0])
          date_max = num2date(times[1])


          if min((date1, date2)) != min((date_min, date_max)):
            print(F"/!\\\n{72*'='}")
            print(date1, date2)
            print(F"{date_min} (x{w_min:4.2f}), {date_max} (x{w_max:4.2f})")
          if max((date1, date2)) != max((date_min, date_max)):
            print(F"/!\\\n{72*'='}")
            print(date1, date2)
            print(F"{date_min} (x{w_min:4.2f}), {date_max} (x{w_max:4.2f})")
            # print(date_min, date_max)

          f.write(
            F"{lon:7.2f}  "
            F"({imin1:02d}, {imin2:02d})  "
            F"({deltaT[imin1]:4.2f}, {deltaT[imin2]:4.2f})  =>  "
            F"({w1:4.2f} * {date1:%Y-%m-%d_%Hh}) + "
            F"({w2:4.2f} * {date2:%Y-%m-%d_%Hh})   "
            # F"{t_min:%Y-%m-%d_%Hh} < "
            # F"{dt_utc} < "
            # F"{t_max:%Y-%m-%d_%Hh}  "
            # F"{(dt_utc - t_min).total_seconds()} "
            # F"{(t_max - dt_utc).total_seconds()}  "
            F"({w_min:4.2f} * {date_min:%Y-%m-%d_%Hh}) + "
            F"({w_max:4.2f} * {date_max:%Y-%m-%d_%Hh})   "
            # F"{w_min:4.2f} {w_max:4.2f} "
            # F"{}"
            # F"{}"
            F"\n"
          )




  print("\n"+72*"=")
  print(f"Run ended in {dt.datetime.now() - run_deb}")






  exit()


  print("\n"+72*"=")
  print(f"Run ended in {dt.datetime.now() - run_deb}")

  exit()
