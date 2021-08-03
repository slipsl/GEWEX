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
import pprint


import numpy as np
import pandas as pd
# from fortio import FortranFile
from scipy.io import FortranFile
from netCDF4 import Dataset

pp = pprint.PrettyPrinter(indent=2)

# Standard library imports
# ========================
# from subroutines import *


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
def freemem():

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
def num2date(val):

  return dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(val))


# #----------------------------------------------------------------------
# def date2num(val):

#   return dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(val))


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
def get_variable(varname, date_curr):

  date_prev = date_curr - dt.timedelta(days=1)
  date_next = date_curr + dt.timedelta(days=1)

  # Get data from the target day, the day before and the day after
  timestep = 1        # in hours
  tstep_per_day = 24  # 

  print(read_var_info(get_filein(varname, date_curr), varname))



  filelist = []

  if date_prev.month < date_curr.month:
    offset = (date_prev.day - 1) * tstep_per_day
    nb_steps = 1 * tstep_per_day
    # print("Filein: ", get_filein(varname, date_prev), offset, nb_steps)
    filelist.append((get_filein(varname, date_prev), offset, nb_steps))

    offset = (date_curr.day - 1) * tstep_per_day
    nb_steps = 2 * tstep_per_day
    # print("Filein: ", get_filein(varname, date_curr), offset, nb_steps)
    filelist.append((get_filein(varname, date_curr), offset, nb_steps))
  elif date_next.month > date_curr.month:
    offset = (date_curr.day - 2) * tstep_per_day
    nb_steps = 2 * tstep_per_day
    # print("Filein: ", get_filein(varname, date_curr), offset, nb_steps)
    filelist.append((get_filein(varname, date_curr), offset, nb_steps))

    offset = (date_next.day - 1) * tstep_per_day
    nb_steps = 1 * tstep_per_day
    # print("Filein: ", get_filein(varname, date_next), offset, nb_steps)
    filelist.append((get_filein(varname, date_next), offset, nb_steps))
  else:
    offset = (date_curr.day - 2) * tstep_per_day
    nb_steps = 3 * tstep_per_day

    # print("Filein: ", get_filein(varname, date_curr), offset, nb_steps)
    filelist.append((get_filein(varname, date_curr), offset, nb_steps))

  # varvalues = np.empty()
  varvalues = []

  for (filename, offset, nb_steps) in filelist:
    print(filename, offset, nb_steps)
    # print(read_netcdf(filename, varname, offset, nb_steps))

    varvalues.extend(read_netcdf(filename, varname, offset, nb_steps))

  varvalues = np.array(varvalues)
  pp.pprint(
    # [num2date(i) for i in varvalues]
    varvalues
  )
  print(varvalues.dtype)

    # offsets = [
    #   (date_curr.day - i) * 24 for i in [2, 1, 0]
    # ]
    # print(offsets)
    # nb_steps = 24

    # times = []
    # for o in offsets:
    #   times.extend(nc_time[o:o+nb_steps])


#----------------------------------------------------------------------
def def_time_lon():

  univT = [i - 24. + 0.5 for i in range(72)]
  read_netcdf(filename, "longitude", offset=None, nb_steps=None)


#----------------------------------------------------------------------
def lon_time(lon, lt_instru):

  l = lon
  if lon > 180.:
    l = lon - 360.
  l = l / 15.

  print("shift = 0.5")
  # univT = [i - 24. for i in range(72)]
  univT = [i - 24. + 0.5 for i in range(72)]
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
def read_var_info(filename, varname):

  print(F"Info from {filename}\n"+80*"=")

  # varname = "time"
  with Dataset(filename, "r", format="NETCDF4") as f_in:
    varout = f_in.variables[varname]

    return (
      varout.name,
      varout.shape,
      varout.units,
      varout.dtype,
    )


#----------------------------------------------------------------------
def def_slice(
  cnt_tim,   cnt_lat,   cnt_lon,   cnt_lev=0,
  off_tim=0, off_lat=0, off_lon=0, off_lev=0,
  stp_tim=None, stp_lat=None, stp_lon=None, stp_lev=None,
):

  if cnt_lev:
    ret = [
      slice(off_tim, off_tim + cnt_tim, stp_tim),
      slice(off_lev, off_lev + cnt_lev, stp_lev),
      slice(off_lat, off_lat + cnt_lat, stp_lat),
      slice(off_lon, off_lon + cnt_lon, stp_lon),
    ]
  else:
    ret = [
      slice(off_tim, off_tim + cnt_tim, stp_tim),
      slice(off_lat, off_lat + cnt_lat, stp_lat),
      slice(off_lon, off_lon + cnt_lon, stp_lon),
    ]

  return ret


#----------------------------------------------------------------------
def read_netcdf(fileid, varname, var_slice):

  return np.squeeze(fileid.variables[varname][var_slice])


#----------------------------------------------------------------------
def read_ERA5_netcdf(date_curr, lt_instru, varname):

  date_prev = date_curr - dt.timedelta(days=1)
  date_next = date_curr + dt.timedelta(days=1)

  file_prev = None
  file_curr = get_filein(varname, date_curr)
  file_next = None

  if date_prev.month < date_curr.month:
    file_prev = get_filein(varname, date_prev)
    # off_prev = (date_prev.day - 1) * 24

  if date_next.month > date_curr.month:
    file_next = get_filein(varname, date_next)
    # off_next = (date_next.day - 1) * 24

  for filename in [file_prev, file_curr, file_next]:
    if filename and not os.path.isfile(filename):
        print(F"Input file missing: {filename}")
        return None

  off_prev, cnt_prev = (date_prev.day - 1) * 24, 24
  off_curr, cnt_curr = (date_curr.day - 1) * 24, 24
  off_next, cnt_next = (date_next.day - 1) * 24, 24

  # Longitudes => timesteps
  print(
    F"{80*'='}\n"
    F"Reading {os.path.basename(file_curr)} to process t = f(lon)"
    F"\n{80*'-'}"
  )
  with Dataset(file_curr, "r", format="NETCDF4") as f_in:
    print(len(f_in.dimensions))
    dims   = f_in.variables[varname].dimensions
    ndim   = f_in.variables[varname].ndim
    nc_lat = f_in.variables["latitude"][:]
    nc_lon = f_in.variables["longitude"][:]
    if "level" in dims:
      nc_lev = f_in.variables["level"][:]
      nlev = nc_lev.size
    else:
      nc_lev = [None, ]
      nlev = 1

    # nc_time = f_in.variables["time"][:]
    ntim = f_in.variables["time"].size
    nlat = nc_lat.size
    nlon = nc_lon.size

  # To get -180. < lon < +180.
  cond = nc_lon[:] > 180.
  nc_lon[cond] = nc_lon[cond] - 360.

  idx_lon_l = np.empty([nlon, ], dtype=int)
  idx_lon_r = np.empty([nlon, ], dtype=int)
  weight_l  = np.empty([nlon, ], dtype=float)
  weight_r  = np.empty([nlon, ], dtype=float)

  # Three days worth of timesteps (ts = 1 hour)
  # univT = [i - 24. + 0.5 for i in range(72)]

  for idx, lon in enumerate(nc_lon):
    # print(lon)

    # localT = univT + lon / 15.
    # deltaT = abs(localT - lt_instru)
    deltaT = [abs((i - 24.+ 0.5 + lon/15.) - lt_instru) for i in range(72)]

    (imin1, imin2) = np.argsort(deltaT)[0:2]

    w1 = deltaT[imin1] / (deltaT[imin1] + deltaT[imin2])
    w2 = deltaT[imin2] / (deltaT[imin1] + deltaT[imin2])

    idx_lon_l[idx] = imin1
    idx_lon_r[idx] = imin2
    weight_l[idx]  = w2
    weight_r[idx]  = w1


  fcurr_in = Dataset(file_curr, "r", format="NETCDF4")
  if file_prev:
    fprev_in = Dataset(file_prev, "r", format="NETCDF4")
  else:
    fprev_in = None
  if file_next:
    fnext_in = Dataset(file_next, "r", format="NETCDF4")
  else:
    fnext_in = None


  # if nc_lev:
  # if "level" in dims:
  # var_full = np.empty([nlev, nlat, nlon], dtype=float)
  var_out = np.empty([nlev, nlat, nlon], dtype=float)
  print("var_out: ", var_out.shape)

  print("loop over levels")
  for idx_pl, pl in enumerate(nc_lev):
    # print(idx_pl, pl)
    print(F"P({idx_pl}) = {pl}mbar")

    if nlev > 1:
      cnt_lev = 1
      off_lev = idx_pl
    else:
      cnt_lev = 0
      off_lev = 0

    # print("File prev")
    if fprev_in:
      f_in = fprev_in
    else:
      f_in = fcurr_in
    var_prev = read_netcdf(
      f_in, varname, 
      def_slice(
        cnt_tim=cnt_prev, off_tim=off_prev,
        cnt_lev=cnt_lev, off_lev=off_lev,
        cnt_lat=nlat, cnt_lon=nlon,
      )
    )
    # print(var_prev.shape)

    # print("File curr")  # time, level, lat, lon
    var_curr = read_netcdf(
      fcurr_in, varname, 
      def_slice(
        cnt_tim=cnt_curr, off_tim=off_curr,
        cnt_lev=cnt_lev, off_lev=off_lev,
        cnt_lat=nlat, cnt_lon=nlon,
      )
    )
    # print(var_curr.shape)

    # print("File next")
    if fnext_in:
      f_in = fnext_in
    else:
      f_in = fcurr_in
    var_next = read_netcdf(
      f_in, varname,
      def_slice(
        cnt_tim=cnt_next, off_tim=off_next,
        cnt_lev=cnt_lev, off_lev=off_lev,
        cnt_lat=nlat, cnt_lon=nlon,
      )
    )
    # print(var_next.shape)

    # freemem()

    var_full = np.concatenate((var_prev, var_curr, var_next), axis = 0)
    # print("var_full: ", var_full.shape)
    # print("var_full: ", var_full[24, 360, 720])

    # freemem()
    # Delete intermediate variables to free some memory
    del var_prev, var_curr, var_next
    # freemem()

    for idx_lon in range(nlon):
      # var_full = [time, lat, lon]
      var_out[idx_pl, :, idx_lon] = (
        var_full[idx_lon_l[idx_lon], :, idx_lon] * weight_l[idx_lon] +
        var_full[idx_lon_r[idx_lon], :, idx_lon] * weight_r[idx_lon]
      )

  # print(var_out[:, 360, 720])

    # print("in : ", var_full[idx_lon_l[idx_lon], :, idx_lon].shape)
    # print("out: ", var_out_pl.shape)


  fcurr_in.close()
  if fprev_in:
    fprev_in.close()
  if fnext_in:
    fnext_in.close()

  sorted_lat_idx = nc_lat.argsort()
  sorted_lon_idx = nc_lon.argsort()

  var_out = var_out[:, sorted_lat_idx, :]
  var_out = var_out[:, :, sorted_lon_idx]

  return np.squeeze(var_out)
  # return np.squeeze(var_out)


    # # nc_var = f_in.variables[varname][off_curr:off_curr+24, :, :, :]
    # nc_var = f_in.variables[varname]
    # ndim = f_in.variables[varname].ndim
    # dims = f_in.variables[varname].dimensions
    # shape = f_in.variables[varname].shape

    # if "level" in dims:
    #   print(dims.index("level"))
    # if "time" in dims:
    #   print(dims.index("time"))

    # var_slice = []
    # for dim, length in zip(dims, shape):
    #   print(dim, length)
    #   if dim == "time":
    #     var_slice.append(slice(off_curr, off_curr + cnt_curr))
    #   elif dim == "level":
    #     var_slice.append(slice(0, 1))
    #   else:
    #     var_slice.append(slice(length))
    # pp.pprint(var_slice)

    # var_values = f_in.variables[varname][var_slice]

    # print(ndim, dims, shape)

    # print(
    #   var_values.shape,
    #   np.squeeze(var_values).shape, 
    #   type(var_values),
    # )


#######################################################################

if __name__ == "__main__":

  run_deb = dt.datetime.now()

  # gives a single float value
  print(psutil.cpu_percent())
  # gives an object with many fields
  print(psutil.virtual_memory())
  # you can convert that object to a dictionary 
  print(dict(psutil.virtual_memory()._asdict()))
  # you can have the percentage of used RAM
  print(psutil.virtual_memory().percent)
  # you can calculate percentage of available memory
  print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)



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
  P_tigr = [
      69.71,
      86.07,
     106.27,
     131.20,
     161.99,
     200.00,
     222.65,
     247.87,
     275.95,
     307.20,
     341.99,
     380.73,
     423.85,
     471.86,
     525.00,
     584.80,
     651.04,
     724.78,
     800.00,
     848.69,
     900.33,
     955.12,
    1013.00,
  ]

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


  fg_temp  = False
  fg_press = True
  fg_h2o   = False

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
    os.path.join(project_dir, "output")
  )
  # dirout   = os.path.normpath(
  #     "/data/slipsl/GEWEX/"
  # )

  if args.verbose:
    print("dirin_pl: ", dirin_pl)
    print("dirin_sf: ", dirin_sf)
    print("dirout  : ", dirout)


  # .. Main program ..
  # ==================

  a = np.arange(6).reshape(2,3)
  print(a)
  print(a.shape)

  it = np.nditer(a, flags=['f_index'])
  for x in it:
    print(F"{x} <{it.index}>")

  it = np.nditer(a, flags=['multi_index'])
  for x in it:
    # print("%d <%s>" % (x, it.multi_index), end=' ')
    print(F"{x} <{it.multi_index}>")

  date_curr = args.date_start

  delta = 1 + (args.date_end - args.date_start).days

  pp.pprint(
    [args.date_start + dt.timedelta(days=i) for i in range(delta)]
  )

  iter_dates = (args.date_start + dt.timedelta(days=i) for i in range(delta))

  # while date_curr <= args.date_end:
  for date_curr in iter_dates:
    date_prev = date_curr - dt.timedelta(days=1)
    date_next = date_curr + dt.timedelta(days=1)

    print(
      F"{date_prev} < {date_curr} > {date_next}"
    )

    fg_process = True


    # Define file names
    pathout = Path(
      os.path.join(
        dirout,
        F"{date_curr:%Y}",
        F"{date_curr:%m}",
      )
    )


    # Path(os.path.join('test_dir', 'level_1b', 'level_2b', 'level_3b')).mkdir(parents=True)

    # Don’t forget these are all flags for the same function. In other words, we can use both exist_ok and parents flags at the same time!

    # if not os.path.isdir(pathout):
    if not pathout.exists():
      print(F"Create output subdirectory: {pathout}")
      pathout.mkdir(parents=True, exist_ok=True)

    # for var in outstr.values():
    #   print(get_fileout(var, date_curr))
    #   filepath = os.path.join(pathout, get_fileout(var, date_curr))
    #   print(filepath)
    #   # Check if output exists
    #   if os.path.isfile(filepath):
    #     print(F"Output file exists. Please remove it and relaunch\n  {filepath}")
    #     fg_process = False

    if fg_process:

      freemem()

      if fg_temp:
        Tpl = read_ERA5_netcdf(date_curr, lt_instru, "ta")
        if not Tpl:
          print(F"Missing data, skip date")
          break
        print(Tpl.shape)
        print(
          F"{Tpl.min():7.2f}K {Tpl.max():7.2f}K {Tpl.mean():7.2f}K"
        )

      freemem()

      if fg_temp:
        Tsurf = read_ERA5_netcdf(date_curr, lt_instru, "skt")
        if not Tsurf:
          print(F"Missing data, skip date")
          break
        print(Tsurf.shape)
        print(
          F"{Tsurf.min():7.2f}K {Tsurf.max():7.2f}K {Tsurf.mean():7.2f}K"
        )

      freemem()

      if fg_press or fg_temp:
        Psurf = read_ERA5_netcdf(date_curr, lt_instru, "sp")
        if Psurf is None:
          print(F"Missing data, skip date")
          continue
        print(Psurf.shape)
        Psurf = Psurf / 100.

        print(
          F"{Psurf.min():7.2f}hPa {Psurf.max():7.2f}hPa {Psurf.mean():7.2f}hPa"
        )

      freemem()

      # it = np.nditer(Tsurf, flags=["multi_index"])
      # for x in it:
      #   print((x, it.multi_index), end=' ')

      nlat, nlon = Psurf.shape
      print(nlat, nlon)

      nlev = len(P_tigr)

      nc_lev = [
        1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
        225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
        775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
      ]

      if fg_temp:
        status = np.full([nlat, nlon], 90000, dtype=int)
        T_tigr = np.zeros([nlev+2, nlat, nlon], dtype=float)
        for idx_lat in range(nlat):
          for idx_lon in range(nlon):
            # T_tigr = np.empty([nlev+2], dtype=float)
            # T_tigr = np.full([nlev+2], -1., dtype=float)
            # T_tigr = np.full([nlev], -1., dtype=float)
            # T_tigr = -1.
            # print(type(T_tigr))
            T_tigr[:nlev, idx_lat, idx_lon] = np.interp(
              P_tigr, nc_lev, Tpl[:, idx_lat, idx_lon]
            )
            cond = P_tigr > Psurf[idx_lat, idx_lon]
            cond = np.append(cond, [True, True, ])
            # print(cond)
            T_tigr[cond] = Tsurf[idx_lat, idx_lon]
            status[idx_lat, idx_lon] = 10000

            if idx_lat % 60 == 0 and idx_lon % 120 == 0:
              print(F"{idx_lat}/{idx_lon} - Psurf = {Psurf[idx_lat, idx_lon]}")
              # print(Tpl[:, idx_lat, idx_lon])
              print("==>")
              for T, P in zip(T_tigr[:, idx_lat, idx_lon], P_tigr):
                print(F"T({P}) = {T:7.2f}", end=" ; ")
              print(F"T(n+1) = {T_tigr[nlev, idx_lat, idx_lon]:7.2f}", end=" ; ")
              print(F"T(n+2) = {T_tigr[nlev+1, idx_lat, idx_lon]:7.2f}")

        print(
          F"{T_tigr.min():7.2f}K {T_tigr.max():7.2f}K {T_tigr.mean():7.2f}K"
        )

      fileout = os.path.join(pathout, get_fileout("L2_P_surf_daily_average", date_curr))
      print(F"Write output to {fileout}")

      print(Psurf)

      # f_out = FortranFile(fileout, mode="w")
      with FortranFile(fileout, mode="w", header_dtype=">u4") as f:
        f.write_record(Psurf.T.astype(dtype=">f4"))



        # T_tigr.extend(Tsurf[idx_lat, idx_lon], 0.)

          # print(T_tigr)
          # print(
          #   [(x, y) for x, y in zip(Tpl[0:, idx_lat, idx_lon], Tpl[1:, idx_lat, idx_lon])]
          # )
          # print(
          #   [x < y for x, y in zip(Tpl[0:, idx_lat, idx_lon], Tpl[1:, idx_lat, idx_lon])]
          #   # all(x < y for x, y in zip(Tpl, Tpl[1:]))
          #   # [x < y for x, y in zip(Tpl, Tpl[1:])].all()
          # )


        # if idx_lat % 60 == 0 and idx_lon % 60 == 0:
        #   print(F"{idx_lat}/{idx_lon}: {cond}")

    # cond = nc_lon[:] > 180.
    # nc_lon[cond] = nc_lon[cond] - 360.



    # # Define file names
    # print(date_curr.year, date_curr.month)
    # pathout = os.path.join(
    #   dirout,
    #   F"{date_curr:%Y}",
    #   F"{date_curr:%m}",
    # )
    # for var in outstr.values():
    #   print(get_fileout(var, date_curr))
    #   filepath = os.path.join(pathout, get_fileout(var, date_curr))
    #   print(filepath)
    #   # Check if output exists
    #   if os.path.isfile(filepath):
    #     print(F"Output file exists. Please remove it and relaunch\n  {filepath}")
    #     fg_process = False

    # def_time_lon()


    # Read "ta" (temperature, 3D)
    # varname = "ta"
    # variable = get_variable(varname, date_curr)

    # Read "skt" (surface temperature, 2D)
    # varname = "skt"
    # variable = get_variable(varname, date_curr)

    # Read "sp" (surface pressure, 2D)

    # Process temperatures and Psurf

    # Check temperatures

    # Read "q" (specific humidity, 3D)
    # Process "q"
    # Check "q"





    # pp.pprint(
    #   [
    #     get_filein(v, date_curr) for v in pl_vars + sf_vars
    #   ]
    # )

    # read_netcdf(get_filein(sf_vars[0], date_curr))

    # date_curr = date_curr + dt.timedelta(days=1)

  print("\n"+80*"=")
  print(f"Run ended in {dt.datetime.now() - run_deb}")

  exit()

