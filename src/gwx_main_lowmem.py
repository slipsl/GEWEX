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
import pprint

import numpy as np
from scipy.io import FortranFile

pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================
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
    "-f", "--force", action="store_true",
    help="If output files exsist, replace them"
  )
  parser.add_argument(
    "--notemp", action="store_true",
    help="Don't produce temperature files"
  )
  parser.add_argument(
    "--noh2o", action="store_true",
    help="Don't produce spec. humidity files"
  )
  parser.add_argument(
    "--nosurf", action="store_true",
    help="Don't produce surface type files"
  )

  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )

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
    F" / Used = {used2:.2f} {units}"
  )

  return


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
    for Vnc in V.ncvars.values():

      f_list.extend([
        f for f in Vnc.get_ncfiles(
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

    return weight, time_indices, (date_bounds.min(), date_bounds.max())


#----------------------------------------------------------------------
def iter_dates(start, stop):

  delta = 1 + (stop - start).days
  return (start + dt.timedelta(days=i) for i in range(delta))


#----------------------------------------------------------------------
def num2date(val):

  return cf.num2date(
    val,
    units=ncgrid.tunits,
    calendar=ncgrid.calendar,  # 'standard',
    only_use_cftime_datetimes=False,  # True,
    only_use_python_datetimes=True,  # False,
    has_year_zero=None
  )


#----------------------------------------------------------------------
def interp(X, Y):

  from scipy.interpolate import interp1d

  return interp1d(
    x=X,
    y=Y,
    fill_value="extrapolate",
  )


#----------------------------------------------------------------------
def get_pressure(params, P):

  P.tgprofiles[...] = P.ncvars["sp"].ncprofiles[...]

  return


#----------------------------------------------------------------------
def get_temp(params, T, P, ncgrid, tggrid):

  import numpy as np
  from scipy.interpolate import interp1d

  for i in range(ncgrid.nlon):
    for j in range(ncgrid.nlat):
      fg_print = not (i % 60) and not (j % 60) and args.verbose
      if fg_print:
        print(
          F"lon = {ncgrid.lon[i]} ; "
          F"lat = {ncgrid.lat[j]}"
        )

      X = ncgrid.lev
      Y = T.ncvars["ta"].ncprofiles[..., j, i]

      cond = np.full(T.tgprofiles.shape[0], False)
      cond[:tggrid.nlev] = tggrid.lev <= P.tgprofiles[j, i]

      fn = interp(X, Y)

      T.tgprofiles[cond, j, i] = fn(tggrid.lev[cond[:tggrid.nlev]])

      T0 = T.ncvars["skt"].ncprofiles[j, i]
      T.tgprofiles[~cond, j, i] = T0

  cond = (T.tgprofiles < 0.)
  if np.any(cond):
    print(
      F"{72*'='}\n"
      F"= {16*' '} /!\\   Negative temperatures   /!\\ {17*' '} =\n"
      F"= {16*' '} /!\\   - {np.count_nonzero(cond):6d} elements         /!\\ {17*' '} =\n"
      F"{72*'='}"
    )

    T.stprofiles[...] = 10000

  return


#----------------------------------------------------------------------
def get_h2o(params, Q, P, ncgrid, tggrid):

  import numpy as np
  # from scipy.interpolate import interp1d

  for i in range(ncgrid.nlon):
    for j in range(ncgrid.nlat):
      fg_print = not (i % 60) and not (j % 60) and args.verbose
      if fg_print:
        print(
          F"lon = {ncgrid.lon[i]} ; "
          F"lat = {ncgrid.lat[j]}"
        )

      X = ncgrid.lev
      Y = Q.ncvars["q"].ncprofiles[..., j, i]

      cond = np.full(Q.tgprofiles.shape[0], False)
      cond[:tggrid.nlev] = tggrid.lev <= P.tgprofiles[j, i]

      fn = interp(X, Y)

      Q.tgprofiles[cond, j, i] = fn(tggrid.lev[cond[:tggrid.nlev]])

      z = np.squeeze(np.argwhere(cond)[-1])
      Q0 = Q.tgprofiles[z, j, i]
      Q.tgprofiles[~cond, j, i] = Q0

  # print(
  #   np.where(Q.ncvars["q"].ncdata < 0.)
  # )

  # print(
  #   np.where(Q.tgprofiles < 0.)
  # )

  # i, j = (592, 435)
  # print(
  #   Q.ncvars["q"].ncprofiles[..., j, i],
  #   Q.tgprofiles[..., j, i],
  # )

  cond = (Q.tgprofiles < 0.)
  if np.any(cond):
    Q.tgprofiles[cond] = 0.
    print(
      F"{72*'='}\n"
      F"= {15*' '} /!\\   Negative spec. humidity   /!\\ {16*' '} =\n"
      F"= {15*' '} /!\\   - {np.count_nonzero(cond):6d} elements         /!\\ {16*' '} =\n"
      F"{72*'='}"
    )

  return


#----------------------------------------------------------------------
def get_surftype(params, S):

  """
  CIRS surface type:
    - 1 = land
    - 2 = ocean
    - 3 = snow / ice
  """
  surftype_land = 1
  surftype_ocean = 2
  surftype_ice = 3

  S.tgprofiles[...] = surftype_ocean

  # Identify land points
  cond = (
    (~S.ncvars["lsm"].ncprofiles.mask) &
    (S.ncvars["lsm"].ncprofiles >= 0.5)
  )
  S.tgprofiles[cond] = surftype_land

  # Identify ice / snow points
  cond = (
    (S.ncvars["sd"].ncprofiles > params.snowdepth_thresh) |
    (
      (~S.ncvars["ci"].ncprofiles.mask) &
      (S.ncvars["ci"].ncprofiles > 0.5)
    )
  )
  S.tgprofiles[cond] = surftype_ice

  return


#----------------------------------------------------------------------
def write_f77(V, filename, profiles, ncgrid, tggrid):

  values = gwv.grid_nc2tg(profiles, ncgrid, tggrid)

  with FortranFile(filename, mode="w", header_dtype=">u4") as f:
    f.write_record(
      np.rollaxis(values, -1, -2).astype(dtype=">f4")
    )

  return


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

  # if args.runtype == 5:
  #   args.force = True

  # ... Constants ...
  # -----------------
  fileversion = "SL04"

  # ... Files and directories ...
  # -----------------------------
  project_dir = Path(__file__).resolve().parents[1]

  instru = gwp.InstruParam(args.runtype)
  params = gwp.GewexParam(project_dir)

  if args.verbose:
    print(instru)
    print(params)

  fg_temp = not args.notemp
  fg_h2o  = not args.noh2o
  fg_surf = not args.nosurf

  # .. Main program ..
  # ==================

  # ... Initialize things ...
  # -------------------------
  # Grids
  ncgrid = gwn.NCGrid()
  tggrid = gwv.TGGrid()
  # Variables
  P = gwv.VarOut("P", instru, fileversion)
  T = Q = S = None
  if fg_temp:
    T = gwv.VarOut("T", instru, fileversion)
  if fg_h2o:
    Q = gwv.VarOut("Q", instru, fileversion)
  if fg_surf:
    S = gwv.VarOut("S", instru, fileversion)

  V_list = tuple(
    V for V in (P, T, Q, S) if V
  )
  if args.verbose:
    print(F"Process {V_list}")

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
    subdir = P.dirout(params.dirout, date_curr)
    if not subdir.exists():
      if args.verbose:
        print(F"Create output subdirectory: {subdir}")
      subdir.mkdir(parents=True, exist_ok=True)

    # ... Load NetCDF & target grids ...
    # ----------------------------------
    if not ncgrid.loaded:
      if T:
        Vnc = T.ncvars["ta"]
      elif Q:
        Vnc = Q.ncvars["q"]
      else:
        Vnc = P.ncvars["sp"]
      if args.verbose:
        print(
          F"Load grid from "
          F"{Vnc.get_ncfiles(params.dirin, args.date_start)}\n"
          F"{72*'='}"
        )
      ncgrid.load(Vnc.get_ncfiles(params.dirin, args.date_start))

    if not tggrid.loaded:
      tggrid.load()

    # ... Compute f(lon, date) stuff ...
    # ----------------------------------
    weight, time_indices, (date_min, date_max) = \
      get_weight_indices(ncgrid.lon, date_curr, instru.tnode)

    for V in V_list:
      # ... Init arrays for variables data ...
      # --------------------------------------
      V.init_datas(ncgrid, tggrid)

      for Vnc in V.ncvars.values():
        # ... Load netcdf data ...
        # ------------------------
        if args.verbose:
          print(F"{72*'~'}\nLoad nc data for {V.name}[{Vnc.name}]")
        Vnc.ncdata = gwn.load_netcdf(
          Vnc, date_min, date_max, params
        )
        freemem()

        # ... Loop over netcdf longitudes ...
        # -----------------------------------
        for i in range(ncgrid.nlon):
          fg_print = not (i % 60) and args.verbose

          if fg_print:
            print(F"lon = {ncgrid.lon[i]}")

          if fg_print:
            print("Weighted nc mean")
          Vnc.get_wght_mean(i, weight[i], time_indices[i])

        # ... Some cleaning to free memory ...
        # ------------------------------------
        if args.verbose:
          print(F"{72*'~'}\nClear datas")
        Vnc.clear_datas(mode="ncdata")
        freemem()




    # # ... Init arrays for variables data ...
    # # --------------------------------------
    # if args.verbose:
    #   print(F"{72*'~'}\nInit datas")
    # # for V in V_list + (surftype, stat, ):
    # for V in V_list:
    #   V.init_datas(ncgrid, tggrid)
    # freemem()

    # # ... Load netcdf data ...
    # # ------------------------
    # if args.verbose:
    #   code_start = dt.datetime.now()
    # for V in V_list:
    #   for Vnc in V.ncvars.values():
    #     if args.verbose:
    #       print(F"{72*'~'}\nLoad nc data for {V.name}[{Vnc.name}]")
    #     Vnc.ncdata = gwn.load_netcdf(
    #       Vnc, date_min, date_max, params
    #     )
    # if args.verbose:
    #   code_stop = dt.datetime.now()
    #   print(code_stop - code_start)

    # # ... Loop over netcdf longitudes ...
    # # -----------------------------------
    # for i in range(ncgrid.nlon):
    #   fg_print = not (i % 60) and args.verbose

    #   if fg_print:
    #     print(F"lon = {ncgrid.lon[i]}")

    #   if fg_print:
    #     print("Weighted nc mean")
    #   for V in V_list:
    #     for Vnc in V.ncvars.values():
    #       Vnc.get_wght_mean(i, weight[i], time_indices[i])

    # ... Process surface pressure ...
    # --------------------------------
    get_pressure(params, P)
    if args.verbose:
      P.print_values()

    # ... Process temperature ...
    # ---------------------------
    if fg_temp:
      get_temp(params, T, P, ncgrid, tggrid)
      if args.verbose:
        T.print_values()

    # ... Process specific humidity ...
    # ---------------------------------
    if fg_h2o:
      get_h2o(params, Q, P, ncgrid, tggrid)
      if args.verbose:
        Q.print_values()

    # ... Process surface type ...
    # ----------------------------
    if fg_surf:
      get_surftype(params, S)
      if args.verbose:
        S.print_values()

    # ... Write everything to F77 binary files ...
    # --------------------------------------------
    if args.verbose:
      print("Write files")
    for V in V_list:
      fileout = V.pathout(params.dirout, date_curr)
      filestat = V.pathout(params.dirout, date_curr, ftype="status")
      if fileout:
        if args.verbose:
          print(V.name, fileout)
        write_f77(V, fileout, V.tgprofiles, ncgrid, tggrid)
      if filestat:
        if args.verbose:
          print(V.name, filestat)
        write_f77(V, filestat, V.stprofiles, ncgrid, tggrid)

    # ... Some cleaning to free memory ...
    # ------------------------------------
    if args.verbose:
      print(F"{72*'~'}\nClear datas")
    for V in V_list:
      V.clear_datas()
    freemem()

    print(
      F"{72*'-'}\n"
      F"{date_curr:%Y-%m-%d} processed in "
      F"{dt.datetime.now() - date_deb}"
    )

  print(F"\n{72*'='}\nRun ended in {dt.datetime.now() - run_deb}")

  exit()
