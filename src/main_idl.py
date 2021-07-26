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
import datetime as dt
# from cftime import num2date, date2num
import pprint


import numpy as np
import pandas as pd
from fortio import FortranFile
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
def read_netcdf(filename, varname, offset=None, nb_steps=None):

  print(F"Reading {filename}\n"+80*"=")


  # varname = "time"
  with Dataset(filename, "r", format="NETCDF4") as f_in:

    varout = f_in.variables[varname]

    # return varout[offset:offset+nb_steps]


    print(f_in.data_model)
    print(f_in.dimensions)

    # for obj in f_in.dimensions.values():
    #   print(obj)

    for obj in f_in.variables.values():
      print(obj)

    print(f_in.variables[varname][:,:,offset:offset+nb_steps])
    # nc_time = f_in.variables["time"]
    # nc_lon  = f_in.variables["longitude"]

    # offsets = [
    #   (date_curr.day - i) * 24 for i in [2, 1, 0]
    # ]
    # print(offsets)
    # nb_steps = 24

    # times = []
    # for o in offsets:
    #   times.extend(nc_time[o:o+nb_steps])
    # for t in times:
    #   print(dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(t)))
    #   # num2date(times[:],units=times.units,calendar=times.calendar)

    # # for time in nc_time:
    # #   print(
    # #     dt.datetime(1800, 1, 1) + dt.timedelta(hours=float(time))
    # #   )

    # lt_instru = 13.5
    # offset = (date_curr.day - 1) * 24

    # print(80*"-")
    # for lon in nc_lon[::180]:
    #   idx_t1, idx_t2, w_t1, w_t2 = lon_time(lon, lt_instru)

    #   tl = date_curr + dt.timedelta(hours=lon/15.)
    #   t_instru = date_curr + dt.timedelta(hours=lt_instru)
    #   t_cible = t_instru - dt.timedelta(hours=lon/15.)

    #   print(
    #     F"lon: {lon} => "
    #     F"tl = {tl}, "
    #     # F"tu = {lt_instru - lon/15.} "
    #     F"tu = {t_cible} "
    #     F"(lt_instru = {lt_instru})"
    #   )

    #   t_ori = dt.datetime(1800, 1, 1)
    #   t1 = t_ori + dt.timedelta(hours=float(times[idx_t1]))
    #   t2 = t_ori + dt.timedelta(hours=float(times[idx_t2]))
    #   # t1 = t_ori + dt.timedelta(hours=float(times[idx_t1 + offset]))
    #   # t2 = t_ori + dt.timedelta(hours=float(times[idx_t2 + offset]))

    #   print(
    #     F" - t1: {t1:%b %d %Hh} - Weight: {w_t1}\n"
    #     F" - t2: {t2:%b %d %Hh} - Weight: {w_t2}"
    #   )
    #   print(80*"-")


    return [1, 2, 3]


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

  off_prev = (date_prev.day - 1) * 24
  off_curr = (date_curr.day - 1) * 24
  off_next = (date_next.day - 1) * 24

  # Longitudes => timesteps
  print(
    F"Reading {os.path.basename(file_curr)} "
    F"to process t = f(lon)\n"+80*"="
  )
  with Dataset(file_curr, "r", format="NETCDF4") as f_in:
    print(len(f_in.dimensions))
    dims = f_in.variables[varname].dimensions
    ndim = f_in.variables[varname].ndim
    nc_lat  = f_in.variables["latitude"][:]
    nc_lon  = f_in.variables["longitude"][:]
    if "level" in dims:
      nc_lev  = f_in.variables["level"][:]
    else:
      nc_lev = None

    # nc_time = f_in.variables["time"][:]
    # nlat = nc_lat.size
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
    deltaT = [abs((i - 24.+0.5 + lon/15.) - lt_instru) for i in range(72)]

    (imin1, imin2) = np.argsort(deltaT)[0:2]

    w1 = deltaT[imin1] / (deltaT[imin1] + deltaT[imin2])
    w2 = deltaT[imin2] / (deltaT[imin1] + deltaT[imin2])

    idx_lon_l[idx] = imin1
    idx_lon_r[idx] = imin2
    weight_l[idx]  = w1
    weight_r[idx]  = w2



  if "level" in dims:
    print("loop over levels")
    for idx_pl, pl in enumerate(nc_lev):
      print(idx_pl, pl)




  with Dataset(file_curr, "r", format="NETCDF4") as f_in:
    # nc_var = f_in.variables[varname][off_curr:off_curr+24, :, :, :]
    nc_var = f_in.variables[varname]
    ndim = f_in.variables[varname].ndim
    dims = f_in.variables[varname].dimensions
    shape = f_in.variables[varname].shape

    if "level" in dims:
      print(dims.index("level"))
    if "time" in dims:
      print(dims.index("time"))

    var_slice = []
    for dim, length in zip(dims, shape):
      print(dim, length)
      if dim == "time":
        var_slice.append(slice(off_curr, off_curr + 24))
      elif dim == "level":
        var_slice.append(slice(0, 1))
      else:
        var_slice.append(slice(length))
    pp.pprint(var_slice)

    var_values = f_in.variables[varname][var_slice]

    print(ndim, dims, shape)

    print(
      var_values.shape,
      np.squeeze(var_values).shape, 
      type(var_values),
    )


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

  date_curr = args.date_start

  while date_curr <= args.date_end:
    date_prev = date_curr - dt.timedelta(days=1)
    date_next = date_curr + dt.timedelta(days=1)

    print(
      F"{date_prev} < {date_curr} > {date_next}"
    )

    fg_process = True

    read_ERA5_netcdf(date_curr, lt_instru, "ta")

    print(psutil.virtual_memory().percent)


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

    date_curr = date_curr + dt.timedelta(days=1)

  print("\n"+80*"=")
  print(f"Run ended in {dt.datetime.now() - run_deb}")

  exit()

