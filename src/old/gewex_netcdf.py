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
# from __future__ import print_function, unicode_literals, division

# Standard library imports
# ========================
# import psutil
# import os
# from pathlib import Path
# import datetime as dt
# from cftime import num2date, date2num
import pprint


# import pandas as pd
# # from fortio import FortranFile
# from scipy.io import FortranFile
from netCDF4 import Dataset

pp = pprint.PrettyPrinter(indent=2)

# Standard library imports
# ========================
# from subroutines import *


# =====================================================================
# =                             Classes                               =
# =====================================================================
class TGGrid(object):
  # -------------------------------------------------------------------
  def __init__(self):
    self.loaded = False

  # -------------------------------------------------------------------
  def __repr__(self):

    return str(self.loaded)

  # -------------------------------------------------------------------
  def load(self, nc_grid):

    import numpy as np

    self.loaded = True

    # Latitude: [-90 ; +90]
    if nc_grid.lat[0] < nc_grid.lat[-1]:
      self.lat = nc_grid.lat
    else:
      self.lat = np.flip(nc_grid.lat)

    # Longitude: [-180 ; +180[
    if nc_grid.lon[0] < nc_grid.lon[-1]:
      self.lon = nc_grid.lon
    else:
      self.lon = np.flip(nc_grid.lon)

    # print(self.lon)

    if np.all(self.lon >= 0):
      cond = self.lon[:] > 180.
      self.lon[cond] = self.lon[cond] - 360.
      print(self.lon[719])

      # print(
      #   np.argmin(self.lon),
      #   self.lon[np.argmin(self.lon)],
      # )

      idx_lon = np.argmax(self.lon) - 1
      self.lon = np.roll(self.lon, idx_lon)

    # Level: 23 levels
      self.lev = np.ma.array(
        [
           69.71,  86.07, 106.27, 131.20,
          161.99, 200.00, 222.65, 247.87,
          275.95, 307.20, 341.99, 380.73,
          423.85, 471.86, 525.00, 584.80,
          651.04, 724.78, 800.00, 848.69,
          900.33, 955.12, 1013.00,
        ],
        mask=False,
      )


# =====================================================================
class NCGrid(object):
  # -------------------------------------------------------------------
  def __init__(self):
    self.loaded = False

  # -------------------------------------------------------------------
  def __repr__(self):

    if not self.loaded:
      ret = str(self.loaded)
    else:
      ret = (
        F"lat: [ {self.lat[0]} ; {self.lat[-1]} ], "
        F"step = {self.lat[1]-self.lat[0]}, len = {self.nlat}\n"
        F"lon: [ {self.lon[0]} ; {self.lon[-1]} ], "
        F"step = {self.lon[1]-self.lon[0]}, len = {self.nlon}\n"
        F"lev: {self.lev}, len = {self.nlev}"
      )

    return ret

  # -------------------------------------------------------------------
  def load(self, filenc):

    # print(F"Load grid from {filenc}\n"+72*"=")
    self.loaded = True

    with Dataset(filenc, "r", format="NETCDF4") as f_nc:
      # f_nc.variables["latitude"].set_auto_mask(False)
      f_nc.set_auto_mask(False)
      self.lat = f_nc.variables["latitude"][:]
      self.lon = f_nc.variables["longitude"][:]
      self.time = f_nc.variables["time"][:]
      self.calendar = f_nc.variables["time"].calendar
      self.tunits = f_nc.variables["time"].units
      if "level" in f_nc.dimensions:
        self.lev = f_nc.variables["level"][:]
      else:
        self.lev = None

    self.nlat = self.lat.size
    self.nlon = self.lon.size
    self.ntime = self.time.size
    if not self.lev is None:
      self.nlev = self.lev.size
    else:
      self.nlev = None

    # print(type(self.lat))
    # print(type(self.lon))
    # print(self.lat.mask)
    # print(np.ma.getmask(self.lat))


# =====================================================================
# =                            Functions                              =
# =====================================================================
#----------------------------------------------------------------------
def get_ncfile(variable, dirin, date):

  if variable.mode == "2d":
    subdir = "AN_SF"
    vartype = "as1e5"
  elif variable.mode == "3d":
    subdir = "AN_PL"
    vartype = "ap1e5"
  else:
    raise(F"Undefined variable {variable}")

  return (
    dirin.joinpath(
      subdir,
      F"{date:%Y}",
      F"{variable.ncvar}.{date:%Y%m}.{vartype}.GLOBAL_025.nc",
    )
  )

  # ta.202102.ap1e5.GLOBAL_025.nc
  # q.202102.ap1e5.GLOBAL_025.nc

  # skt.202102.as1e5.GLOBAL_025.nc
  # sp.202102.as1e5.GLOBAL_025.nc


#----------------------------------------------------------------------
def read_netcdf_t(variable, t):


  with Dataset(variable.ncfiles, "r", format="NETCDF4") as f:
    rec = f.variables[variable.ncvar][t, ...]

  return (variable.coeff * rec).copy()


#----------------------------------------------------------------------
def read_netcdf(variable, nc_grid, i_lon, i_time):

  import numpy as np
  if isinstance(variable.ncfiles, tuple):
    time_slices = ((i_time[0], ), (i_time[1], ))
    ncfiles = variable.ncfiles
    # print("2 files")
  else:
    time_slices = ((i_time,))
    ncfiles = (variable.ncfiles, )
    # print("1 files")

  if variable.mode == "2d":
    xyz_slice = (slice(nc_grid.nlat), i_lon)
  else:
    xyz_slice = (slice(nc_grid.nlev), slice(nc_grid.nlat), i_lon)

  l_var = []
  for time_slc, filename in zip(time_slices, ncfiles):
    var_slice = (time_slc, ) + xyz_slice
    with Dataset(filename, "r", format="NETCDF4") as f_in:
      l_var.append(f_in.variables[variable.ncvar][var_slice])
      # print(f_in.variables[variable.ncvar].missing_value)
      # print(f_in.variables[variable.ncvar][:].fill_value)
      # print(np.ma.is_masked(f_in.variables[variable.ncvar][:]))

  if len(l_var) > 1:
    var = np.ma.concatenate(l_var, axis=0)
  else:
    var = l_var[0]

  if np.ma.is_masked(var):
    print(
      F"{72*'='}"
      F"=== /!\\   Missing values in {filename}   /!\\"
      F"{72*'='}"
    )
  # print(type(var))

  # exit()

  return (variable.coeff * var).copy()
  # return var


#----------------------------------------------------------------------
def check_miss_val(var, ncfiles, params):

  import numpy as np
  from pathlib import Path

  if np.ma.is_masked(var):
    print(
      F"{72*'='}\n"
      F"= {18*' '} /!\\   Missing values in:   /!\\ {18*' '} ="
      # F"{72*'-'}"
    )
    with open(params.dirlog.joinpath("ERA5_missvalues.dat"), "a") as f:
      for ncf in set(ncfiles):
        print(F"= {ncf}")
        f.write(F"{ncf}\n")
      print(F"{72*'='}")

  return


#----------------------------------------------------------------------
def load_netcdf(V, date_min, date_max, params):

  import numpy as np
  from pathlib import Path

  ncfiles = V.get_ncfiles(params.dirin, (date_min, date_max))

  t_min, t_max = [
    d.hour + 24 * (d.day - 1)
    for d in (date_min, date_max)
  ]

  if V.name == "time":
    ncfiles = tuple(
      Path(str(f).replace("time.", "sp."))
      for f in ncfiles
    )

  with Dataset(ncfiles[0], "r", format="NETCDF4") as f:
    ntim = f.dimensions["time"].size
    if V.ncvar not in f.variables:
      ncvar = V.ncvar_alt
    else:
      ncvar = V.ncvar
    miss_val = f.variables[ncvar].missing_value
    # print(miss_val)

  (t0_i, t1_f) = (t_min, t_max + 1)
  if ncfiles[0] == ncfiles[1]:
    t0_f = t1_i = int((t_min + t_max) / 2)
  else:
    (t0_f, t1_i) = (ntim, 0)

  # print(ncfiles[0], t0_i, t0_f, t0_f - t0_i)
  with Dataset(ncfiles[0], "r", format="NETCDF4") as f:
    var0 = f.variables[ncvar][t0_i:t0_f, ...].copy()
  # if np.ma.is_masked(var0):
  #   print(
  #     F"{72*'='}\n"
  #     F"=== /!\\   Missing values in {ncfiles[0]}   /!\\\n"
  #     F"{72*'='}"
  #   )
  var0 = var0 * V.coeff
  # print(var0.shape)
  # print(ncfiles[1], t1_i, t1_f, t1_f - t1_i)
  with Dataset(ncfiles[1], "r", format="NETCDF4") as f:
    var1 = f.variables[ncvar][t1_i:t1_f, ...].copy()
  var1 = var1 * V.coeff
  # print(var1.shape)

  var = np.ma.concatenate((var0, var1), axis=0)

  check_miss_val(var, ncfiles, params)


  # cond = (var == miss_val)
  # if not cond.all():
  #   print("missing")
  #   print(var[cond])
  # if np.ma.is_masked(var):
  #   print(
  #     F"{72*'='}\n"
  #     F"=== /!\\   Missing values in {ncfiles}   /!\\\n"
  #     F"{72*'='}"
  #   )


  # if V.name == "ci":
  #   print(var)

  # print(type(var))

  return var
