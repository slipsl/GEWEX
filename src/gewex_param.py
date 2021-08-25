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


# import numpy as np
# import pandas as pd
# # from fortio import FortranFile
# from scipy.io import FortranFile
# from netCDF4 import Dataset

pp = pprint.PrettyPrinter(indent=2)

# Standard library imports
# ========================
# from subroutines import *


# =====================================================================
# =                             Classes                               =
# =====================================================================
class InstruParam(object):
  # -------------------------------------------------------------------
  def __init__(self, runtype):

    runtypes = {
      1: {"name": "AIRS_V6", "f_q": 1.e3, "tnode":  1.5, "ampm": "AM"},
      2: {"name": "AIRS_V6", "f_q": 1.e3, "tnode": 13.5, "ampm": "PM"},
      3: {"name": "IASI",    "f_q": 1.,   "tnode":  9.5, "ampm": "AM"},
      4: {"name": "IASI",    "f_q": 1.,   "tnode": 23.5, "ampm": "PM"},
      5: {"name": "TEST",    "f_q": 1.e3, "tnode":  0.0, "ampm": "AM"},
      # 1: {"name": "AIRS_V6", "f_q": 1.e3, "f_p": 100., "tnode":  1.5, "ampm": "AM"},
      # 2: {"name": "AIRS_V6", "f_q": 1.e3, "f_p": 100., "tnode": 13.5, "ampm": "PM"},
      # 3: {"name": "IASI",    "f_q": 1.,   "f_p": 100., "tnode":  9.5, "ampm": "AM"},
      # 4: {"name": "IASI",    "f_q": 1.,   "f_p": 100., "tnode": 23.5, "ampm": "PM"},
    }

    self.name  = runtypes[runtype]["name"]
    self.f_q   = runtypes[runtype]["f_q"]
    self.f_p   = 100.  # runtypes[runtype]["f_p"]
    self.tnode = runtypes[runtype]["tnode"]
    self.ampm  = runtypes[runtype]["ampm"]

  # -------------------------------------------------------------------
  def __repr__(self):
    return (
      F"Instrument:   {self.name}\n"
      F"H2O coeff:    {self.f_q}\n"
      F"P coeff:      {self.f_p}\n"
      F"Time at node: {self.tnode}\n"
      F"AM / PM:      {self.ampm}"
    )


# =====================================================================
class GewexParam(object):
  # -------------------------------------------------------------------
  def __init__(self, project_dir):

    from pathlib import Path
    # import platform
    import socket
 
    self.fileversion = "05"

    # print(platform.node())
    if "ciclad" in socket.gethostname():
      self.dirin = Path("/bdd/ERA5/NETCDF/GLOBAL_025/hourly")
      self.dirout = Path("/data/slipsl/GEWEX/ERA5_averages")
    else:
      self.dirin = project_dir.joinpath("input")
      self.dirout = project_dir.joinpath("output")


  # -------------------------------------------------------------------
  def __repr__(self):
    return (
      F"File version: {self.fileversion}\n"
      F"Input dir:    {self.dirin}\n"
      F"Output dir:   {self.dirout}"
    )


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
      self.longname = ""
      self.ncvar = None
      self.mode = None
      self.coeff = None
      self.str = "L2_status"

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

    self.ncprofiles = np.full(ncshape, np.nan)
    self.tgprofiles = np.full(tgshape, np.nan)


  # -------------------------------------------------------------------
  def init_w_mean(self, grid): 

    import numpy as np

    if self.mode == "2d":
      shape = (grid.nlon, grid.nlat)
    else:
      shape = (grid.nlon, grid.nlat, grid.nlev)

    self.w_mean = np.ma.empty(shape)
    self.w_mean.mask = True

  # -------------------------------------------------------------------
  def init_outval(self, grid): 

    import numpy as np

    if self.mode == "2d":
      shape = (grid.nlon, grid.nlat)
    else:
      if self.name == "temp":
        shape = (grid.nlon, grid.nlat, grid.nlev+2)
      else:
        shape = (grid.nlon, grid.nlat, grid.nlev)

    self.outvalues = np.ma.empty(shape)
    self.outvalues.mask = True

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
      w_max * self.ncdata[t_min, ..., i]
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

    # print(self.tgprofiles.shape[0])

    self.tgprofiles[cond, j, i] = fn(tggrid.lev[cond[:tggrid.nlev]])

    if not V0:
      z = np.squeeze(np.argwhere(cond)[-1])
      V0 = self.tgprofiles[z, j, i]
      # V0 = tgprofile[np.squeeze(np.argwhere(cond)[-1])]

    self.tgprofiles[~cond, j, i] = V0
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

    # return
    # return tgprofile






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
  def load(self, nc_grid):

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


#----------------------------------------------------------------------









"""
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

  print(F"Info from {filename}\n"+72*"=")

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
    F"{72*'='}\n"
    F"Reading {os.path.basename(file_curr)} to process t = f(lon)"
    F"\n{72*'-'}"
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
"""