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
# import psutil
# import os
# from pathlib import Path
# import datetime as dt
# from cftime import num2date, date2num
# import pprint


# import numpy as np
# import pandas as pd
# # from fortio import FortranFile
# from scipy.io import FortranFile
# from netCDF4 import Dataset

# pp = pprint.PrettyPrinter(indent=2)

# Standard library imports
# ========================
# from subroutines import *


# =====================================================================
# =                             Classes                               =
# =====================================================================
class Variable(object):
  # -------------------------------------------------------------------
  def __init__(self, name, instru):

    import numpy as np

    self.name = name

    if name == "Psurf":
      self.longname = "Surface pressure"
      self.ncvar = "sp"
      self.mode = "2d"
      self.coeff = 1.e-2
      self.str = "L2_P_surf_daily_average"
      self.units = "hPa"
    if name == "Tsurf":
      self.longname = "Skin temperature"
      self.ncvar = "skt"
      self.mode = "2d"
      self.coeff = 1.
      self.str = None
      self.units = "K"
    if name == "temp":
      self.longname = "Temperature"
      self.ncvar = "ta"
      self.mode = "3d"
      self.coeff = 1.
      self.str = "L2_temperature_daily_average"
      self.units = "K"
    if name == "h2o":
      self.longname = "Specific humidity"
      self.ncvar = "q"
      self.mode = "3d"
      self.coeff = instru.f_q
      self.str = "L2_H2O_daily_average"
      self.units = F"{self.coeff ** -1:1.0e} kg kg**-1"
    if name == "stat":
      self.longname = "Temp status"
      self.ncvar = None
      self.mode = "2d"
      self.coeff = None
      self.str = "L2_status"

    if name == "time":
      self.longname = "Time"
      self.ncvar = "time"
      self.mode = "2d"
      self.coeff = 1.
      self.str = None

    self.fileversion = "05"
    self.instru = instru.name
    self.ampm = instru.ampm

    self.outvalues = None

  # -------------------------------------------------------------------
  def __bool__(self):
    return True

  # -------------------------------------------------------------------
  def __repr__(self):
    return (
      F"{self.name} ({self.mode}): "
      F"{self.ncvar} * {self.coeff} => "
      F"{self.str}"
    )

  # -------------------------------------------------------------------
  def init_datas(self, ncgrid, tggrid):

    import numpy as np

    if self.mode == "2d":
      ncshape = tgshape = (ncgrid.nlat, ncgrid.nlon)
    else:
      if self.name == "temp":
        tg_nlev = tggrid.nlev + 2
      else:
        tg_nlev = tggrid.nlev
      ncshape = (ncgrid.nlev, ncgrid.nlat, ncgrid.nlon)
      tgshape = (tg_nlev, ncgrid.nlat, ncgrid.nlon)

    print(ncshape, tgshape)

    self.ncprofiles = np.full(ncshape, np.nan)
    self.tgprofiles = np.full(tgshape, np.nan)
    self.ncdata = None

  # -------------------------------------------------------------------
  def clear_datas(self):

    del self.ncprofiles
    del self.tgprofiles
    del self.ncdata

  # -------------------------------------------------------------------
  def fileout(self, date):

    if self.str:
      ret = (
        F"unmasked_ERA5_{self.instru}_{self.str}."
        F"{date:%Y%m%d}."
        F"{self.ampm}_{self.fileversion}"
      )
    else:
      ret = None

    return ret

  # -------------------------------------------------------------------
  def dirout(self, dirout, date):

    if self.str:
      ret = dirout.joinpath(F"{date:%Y}", F"{date:%m}")
    else:
      ret = None

    return ret

  # -------------------------------------------------------------------
  def pathout(self, dirout, date):

    if self.str:
      ret = self.dirout(dirout, date).joinpath(self.fileout(date))
    else:
      ret = None

    return ret

  # -------------------------------------------------------------------
  def get_ncfiles(self, dirin, dates):

    if isinstance(dates, tuple):
      dt_list = dates
    else:
      dt_list = (dates, )

    if self.mode == "2d":
      subdir = "AN_SF"
      vartype = "as1e5"
    elif self.mode == "3d":
      subdir = "AN_PL"
      vartype = "ap1e5"
    else:
      raise(F"Undefined variable {self.name}")

    zone = "GLOBAL"
    resol = "025"

    ret = tuple(
      dirin.joinpath(
        subdir,
        F"{date:%Y}",
        F"{self.ncvar}.{date:%Y%m}.{vartype}.{zone}_{resol}.nc"
      ) 
      for date in dt_list
    )

    if not isinstance(dates, tuple):
      ret = ret[0]

    return ret

  # -------------------------------------------------------------------
  def print_values(self, mode="nc"):

    if mode == "nc":
      values = self.ncvalues
    else:
      values = self.outvalues

    if values is None:
      string = "None"
    else:
      fmt = ".2f"
      string = (
        F"{self.longname} ({mode}): "
        F"min={values.min():{fmt}} {self.units} ; "
        F"max={values.max():{fmt}} {self.units} ; "
        F"mean={values.mean():{fmt}} {self.units} ; "
        F"std={values.std():{fmt}} {self.units}"
      )
    print(string)

  # -------------------------------------------------------------------
  def get_wght_mean(self, i, w, t):

    t_min, t_max = t
    w_min, w_max = w

    self.ncprofiles[..., i] = (
      w_min * self.ncdata[t_min, ..., i] +
      w_max * self.ncdata[t_max, ..., i]
    )

  # -------------------------------------------------------------------
  def get_interp(self, i, j, ncgrid, tggrid, P0, V0):

    import numpy as np
    from scipy.interpolate import interp1d

    X = ncgrid.lev
    Y = self.ncprofiles[..., j, i]

    cond = np.full(self.tgprofiles.shape[0], False)
    cond[:tggrid.nlev] = tggrid.lev <= P0

    fn = interp1d(
      x=X,
      y=Y,
      fill_value="extrapolate",
    )

    self.tgprofiles[cond, j, i] = fn(tggrid.lev[cond[:tggrid.nlev]])

    if not V0:
      z = np.squeeze(np.argwhere(cond)[-1])
      V0 = self.tgprofiles[z, j, i]

    self.tgprofiles[~cond, j, i] = V0


class TGGrid(object):
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
  def load(self):

    import numpy as np

    self.loaded = True

    # Latitude: [-90 ; +90]
    # self.lat = np.ma.array(
    self.lat = np.array(
      np.arange(-90., 90.1, 0.25),
      # mask=False,
    )
    self.nlat = self.lat.size

    # Longitude: [-180 ; +180[
    # self.lon = np.ma.array(
    self.lon = np.array(
      # np.arange(-180., 180., 0.25),
      np.arange(-179.75, 180.1, 0.25),
      # mask=False,
    )
    self.nlon = self.lon.size

    # Level: 23 levels
    # self.lev = np.ma.array(
    self.lev = np.array(
      [
         69.71,  86.07, 106.27, 131.20,
        161.99, 200.00, 222.65, 247.87,
        275.95, 307.20, 341.99, 380.73,
        423.85, 471.86, 525.00, 584.80,
        651.04, 724.78, 800.00, 848.69,
        900.33, 955.12, 1013.00,
      ],
      # mask=False,
    )
    self.nlev = self.lev.size

# =====================================================================
# =                            Functions                              =
# =====================================================================
def grid_nc2tg(var_in, ncgrid, tggrid):
  import numpy as np

  var_out = var_in

  # Longitudes
  #   [0.; 360.[ => ]-180.; 180.]
  # =============================
  # print(ncgrid.lon[0])
  # print(ncgrid.lon[-1])

  # print(tggrid.lon[0])
  # print(tggrid.lon[-1])

  # lon = ncgrid.lon.copy()
  # cond = (ncgrid.lon >= 180.)
  # lon[cond] = lon[cond] - 360.
  # print(lon)

  # # imin = np.argmin(lon)
  # # print(imin, lon[imin])
  # # print(np.roll(lon, -imin))
  # # print(np.roll(ncgrid.lon, -imin))
  # # print(tggrid.lon)

  # imin = np.squeeze(np.argwhere(lon == tggrid.lon[0]))
  # print("imin", imin)
  # print(
  #   np.roll(
  #     # lon,
  #     ncgrid.lon,
  #     -imin
  #   )
  # )

  lon_init = tggrid.lon[0]
  if lon_init < 0.:
    lon_init = lon_init + 360.
  imin = np.squeeze(np.argwhere(ncgrid.lon == lon_init))

  # print(
  #   "imin",
  #   imin,
  #   np.roll(ncgrid.lon, -(imin))
  # )

  # l = np.roll(ncgrid.lon, -(imin))
  # cond = l > 180.
  # l[cond] = l[cond] - 360.
  # pp.pprint(l)

  # var_out = np.roll(var_out, -(imin-1), axis=-1)
  var_out = np.roll(var_out, -imin, axis=-1)

  # Latitudes
  #   [+90.; -90.] => [-90.; +90.]
  # ==============================
  var_out = np.flip(var_out, axis=-2)

  return var_out.copy()
