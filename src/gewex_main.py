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
def compute_outdata(variable, w_min, w_max, grid, fg_print):

  if variable.mode == "2d":
    shape = (grid.nlat, )
  else:
    shape = (grid.nlev, grid.nlat, )
  outvalues = np.empty(shape)

  if fg_print:
    print(F"inside compute: {variable.ncvalues.shape} => {outvalues.shape}")

  outvalues = (
    w_min * variable.ncvalues[0, ...] +
    w_max * variable.ncvalues[1, ...]
  )
  if fg_print:
    print("sortie de compute:", type(outvalues), outvalues.shape)

  return outvalues


#----------------------------------------------------------------------
def iter_dates(start, stop):

  # # a generator that yields items instead of returning a list
  # def firstn(n):
  #     num = 0
  #     while num < n:
  #         yield num
  #         num += 1

  delta = 1 + (stop - start).days

  # pp.pprint(
  #   [args.date_start + dt.timedelta(days=i) for i in range(delta)]
  # )

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


#----------------------------------------------------------------------


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

  # print(type(args.date_start))
  # print(type(args.date_end))

  # date_start = dt.datetime.strptime(args.date_start, "%Y%m%d")
  # date_end = dt.datetime.strptime(args.date_end, "%Y%m%d")

  # print("year:  ", args.date_start.year)
  # print("month: ", args.date_start.month)

  # ... Constants ...
  # -----------------
  # P_tigr = [
  #     69.71,  86.07, 106.27, 131.20,
  #    161.99, 200.00, 222.65, 247.87,
  #    275.95, 307.20, 341.99, 380.73,
  #    423.85, 471.86, 525.00, 584.80,
  #    651.04, 724.78, 800.00, 848.69,
  #    900.33, 955.12, 1013.00,
  # ]


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
  fg_temp  = False
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

    # nstep = 24  # number of time steps per day
    # nday  = 3
    # deb = (date_prev.day - 1) * nstep
    # fin = (date_prev.day - 1) * nstep + nday * nstep
    # times = nc_grid.time[deb:fin]
    # pp.pprint(num2date(times))

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
        if variable:
          # Find which NetCDF file(s) contain wanted date(s)
          if fg_print:
            print(F"Find NetCDF files to read")
          variable.ncfiles = variable.get_ncfiles(params.dirin, dates)
          # pp.pprint(variable.ncfiles)
          # initialize output variables
          if variable.outvalues is None:
            variable.init_outval(tg_grid)

          # Read input data
          if fg_print:
            print(F"Read {variable.name}, lon = {lon} ({i_lon})")
          variable.ncvalues = gnc.read_netcdf(
            variable, nc_grid, i_lon, (t_min, t_max)
          )
          if fg_print:
            variable.print_values()

          # Compute output data
          outvalues = compute_outdata(variable, w_min, w_max, nc_grid, fg_print)
          # print(outvalues.shape)
          if variable.mode == "2d":
            variable.outvalues[..., i_lon] = outvalues
            if fg_print:
              variable.print_values("out")
          else:
            # if fg_print:
            #   print(
            #     F"3d: {outvalues.shape} "
            #     F"{variable.outvalues[..., i_lon].shape} "
            #     F"{outvalues.min()} {outvalues.max()} "
            #   )
            if fg_print:
              print(F"Interpolate")

              print(outvalues.shape)

              ftest = interp1d(
                x=nc_grid.lev,
                y=outvalues[:, :],
                axis=0,
                fill_value="extrapolate",
              )

              print(ftest(tg_grid.lev.getdata))




              # levels = np.tile(tg_grid.lev, (tg_grid.nlat, 1)).T

              # # shape = (10, 1)
              # # print(
              # #   # np.tile(tg_grid.lev, shape).T.shape,
              # #   # np.tile(tg_grid.lev, shape).T
              # #   levels.shape,
              # #   levels,
              # # )
              # # cond = tg_grid.lev > Psurf.outvalues[:, i_lon]
              # cond = levels <= Psurf.outvalues[:, i_lon]

              # i_lat = 700
              # # # pp.pprint(Psurf.outvalues[i_lat, i_lon])
              # # pp.pprint(outvalues[:, i_lat])
              # # # pp.pprint(levels[:, i_lat])
              # # # pp.pprint(cond[:, i_lat])
              # # # pp.pprint(cond)



              # print(
              #   np.diff(outvalues[..., i_lat])
              # )



              # # print(np.interp(
              # #   tg_grid.lev,
              # #   nc_grid.lev,
              # #   outvalues[..., i_lat]
              # # ))



              # # interp_val = np.ma.empty((tg_grid.nlev, tg_grid.nlat))
              # # interp_val.mask = True



              # # interp_val[cond] = np.interp(
              # #   tg_grid.lev,
              # #   nc_grid.lev,
              # #   outvalues
              # # )

              # # # T_tigr[:idx_Pmax+1, idx_lat, idx_lon] = np.interp(
              # # #   P_tigr[:idx_Pmax+1],
              # # #   nc_lev, Tpl[:, idx_lat, idx_lon]
              # # # )


              exit()




      continue


      # if fg_print:
      #   print(F"Read temp (3d), lon = {lon} ({i_lon})")
      # T.ncvalues = gnc.read_netcdf(T, nc_grid, i_lon, (t_min, t_max))
      # if fg_print:
      #   print(
      #     F"Temp : "
      #     F"{T.ncvalues.min():7.2f} K {T.ncvalues.max():7.2f} K"
      #     F"{T.ncvalues.mean():7.2f }K {T.ncvalues.std():7.2f} K"
      #   )

      # slc_time = ([t_min, ], [t_max, ], )
      # for variable in (Psurf, Tsurf, T, Q):
      #   variable.ncfiles = (
      #     gnc.get_ncfile(variable, params.dirin, dt_min),
      #     gnc.get_ncfile(variable, params.dirin, dt_max),
      #   )

      # Psurf
      # =====
      vars = []
      for (idx_time, filenc) in zip(slc_time, Psurf.ncfiles):
        if fg_print:
          print(idx_time, filenc)
        with Dataset(filenc, "r", format="NETCDF4") as f_in:
          var = f_in.variables[Psurf.ncvar][idx_time, :, i_lon].copy()
          vars.append(var)
        # if fg_print:
        #   print(
        #     # f_in.variables[Psurf.ncvar][idx_time, :, lon].shape
        #     var,
        #     var.shape,
        #     type(var),
        #       )
      if len(vars) > 1:
        var = np.ma.concatenate(vars, axis=0)
      if fg_print:
        print(
          # f_in.variables[Psurf.ncvar][idx_time, :, lon].shape
          # var,
          var.shape,
          type(var),
            )

      # # Psurf
      # # =====
      # with Dataset(Psurf.ncfiles[0], "r", format="NETCDF4") as f_in:
      #   var_min = f_in.variables[Psurf.ncvar][(t_min, ), :, lon].copy()
      # with Dataset(Psurf.ncfiles[1], "r", format="NETCDF4") as f_in:
      #   var_max = f_in.variables[Psurf.ncvar][(t_max, ), :, lon].copy()

      # var = np.ma.concatenate([var_min, var_max], axis=0)
      # if fg_print:
      #   print(
      #     # var,
      #     var_min.shape,
      #     var_max.shape,
      #     var.shape,
      #     type(var),
      #   )

      # for (idx_time, filenc) in zip(slc_time, Psurf.ncfiles):
      #   if fg_print:
      #     print(idx_time, filenc)
      #   with Dataset(filenc, "r", format="NETCDF4") as f_in:
      #       var = f_in.variables[Psurf.ncvar][idx_time, :, lon].copy()
      #   if fg_print:
      #     print(
      #       # f_in.variables[Psurf.ncvar][idx_time, :, lon].shape
      #       var,
      #       var.shape,
      #       type(var),
      #         )

      # # temp
      # # =====
      # for (idx_time, filenc) in zip(slc_time, T.ncfiles):
      #   if fg_print:
      #     print(idx_time, filenc)
      #   with Dataset(filenc, "r", format="NETCDF4") as f_in:
      #     # view = f_in.variables[T.ncvar][idx_time, :, :, lon]
      #     var = f_in.variables[T.ncvar][idx_time, :, :, lon].copy()
      #   if fg_print:
      #     print(
      #       # f_in.variables[T.ncvar][idx_time, :, :, lon].shape
      #       # # type(view)
      #       var,
      #       var.shape,
      #       type(var),
      #     )



      #  # Psurf.ncvalues = 

      # (fnc_min, fnc_max) = (
      #   gnc.get_ncfile(Psurf, params.dirin, date) for date in (dt_min, dt_max)
      # )


      # if fg_print:
      #   # print(fnc_min)
      #   # print(fnc_max)
      #   print(Q.ncfiles[0])
      #   print(slc_time[0])




    print(Psurf.outvalues[180, :])

    values = np.roll(Psurf.outvalues, -721, axis=-1)
    values = np.flip(values, axis=-2)

    # f_out = FortranFile(fileout, mode="w")
    # with FortranFile("Ptest.dat", mode="w", header_dtype=">u4") as f:
    with FortranFile(
      Psurf.pathout(params.dirout, date_curr), mode="w",
      header_dtype=">u4"
    ) as f:
      f.write_record(values.T.astype(dtype=">f4"))







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









  exit()


  print("\n"+72*"=")
  print(f"Run ended in {dt.datetime.now() - run_deb}")

  exit()
