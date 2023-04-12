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
from pathlib import Path
import datetime as dt
# from cftime import num2date, date2num
# import pprint


import numpy as np
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
class VarNC(object):
  # -------------------------------------------------------------------
  def __init__(self, name, mode, coeff, altname=None, valid_range=None):

    self.name = name
    self.altname = altname
    self.mode = mode
    self.coeff = coeff
    self.valid_range = valid_range

  # -------------------------------------------------------------------
  def __repr__(self):
    return (
      F"{self.name} ({self.mode})"
    )

  # -------------------------------------------------------------------
  def init_datas(self, ncgrid):

    # import numpy as np

    if self.mode == "2d":
      ncshape = (ncgrid.nlat, ncgrid.nlon)
    else:
      ncshape = (ncgrid.nlev, ncgrid.nlat, ncgrid.nlon)

    self.ncprofiles = np.ma.empty(ncshape)
    self.ncdata = None

    # if self.mode == "2d":
    #   ncshape = tgshape = (ncgrid.nlat, ncgrid.nlon)
    # else:
    #   if self.name == "temp":
    #     tg_nlev = tggrid.nlev + 2
    #   else:
    #     tg_nlev = tggrid.nlev
    #   ncshape = (ncgrid.nlev, ncgrid.nlat, ncgrid.nlon)
    #   tgshape = (tg_nlev, ncgrid.nlat, ncgrid.nlon)

    # print(ncshape, tgshape)

    # # self.ncprofiles = np.full(ncshape, np.nan)
    # # self.tgprofiles = np.full(tgshape, np.nan)
    # self.ncprofiles = np.ma.empty(ncshape)
    # self.tgprofiles = np.ma.empty(tgshape)
    # self.ncdata = None

  # -------------------------------------------------------------------
  def clear_datas(self, mode="full"):

    if mode == "full":
      del self.ncprofiles
    try:
      del self.ncdata
    except AttributeError as err:
      pass

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
        F"{self.name}.{date:%Y%m}.{vartype}.{zone}_{resol}.nc"
      ) 
      for date in dt_list
    )

    if not isinstance(dates, tuple):
      ret = ret[0]

    return ret

  # -------------------------------------------------------------------
  def get_wght_mean(self, i, w, t):

    t_min, t_max = t
    w_min, w_max = w

    self.ncprofiles[..., i] = (
      w_min * self.ncdata[t_min, ..., i] +
      w_max * self.ncdata[t_max, ..., i]
    )


  # -------------------------------------------------------------------
  def check_range(self, i, w, t, date_min, nclev, dirdata, args):

    import numpy as np
    import datetime as dt

    fg_print = i == 513  # and j == 456  # and args.verbose

    t_min, t_max = t
    w_min, w_max = w

    # if fg_print:
    #   print(self.ncdata.ndim)
    #   if self.ncdata.ndim == 4:
    #     j = 456
    #     for a, b in zip(self.ncdata[t_min, ..., j, i], self.ncdata[t_max, ..., j, i]):
    #       print(F"{a:8.5f}, {b:8.5f}")

    if self.valid_range:
      vmin, vmax = (x * self.coeff for x in self.valid_range)

      cond = np.logical_or(
        np.logical_or(
          self.ncdata[t_min, ..., i] > vmax,
          self.ncdata[t_min, ..., i] < vmin,
        ),
        np.logical_or(
          self.ncdata[t_max, ..., i] > vmax,
          self.ncdata[t_max, ..., i] < vmin,
        ),
      )
      if np.any(cond):
        filename = Path(
          F"Invalid_{self.name}_r{args.runtype}_"
          F"{args.date_start:%Y%m%d}_{args.date_end:%Y%m%d}_"
          F"{args.version}.dat"
        )
        filepath = dirdata.joinpath(filename)

        if filepath.is_file():
          write_header = False
        else:
          write_header = True

        # print(
        #   filepath,
        #   filepath.is_file()
        # )

        print(vmin, vmax)
        print(t_min, t_max)
        arg_array = np.argwhere(cond).copy()
        arg_array = arg_array[np.lexsort(
          (arg_array[:, -1], arg_array[:, -2])
        )]

        with open(filepath, "a") as f:
          if write_header:
            f.write(
              # F"#    i   j   l t1 t2  "
              F"# {'i':>4} {'j':>3} {'l':>3} t1 t2  "
              F"{' lon':7} {'lat':6} {' lev':7} "
              F"{'date1':12} {'date2':12} "
              F"{'weight1':7} {'weight2':7} "
              F"{' val1':12} {' val2':12} "
              F"{' vmin':9} {' vmax':9} "
              F"\n# {130*'='}\n"
            )

          for idx in arg_array:
            # print(
            #   idx,
            #   # ncdata[tuple(idx)]
            # )
            lon = 0.25 * i
            lat = 90 - (0.25 * idx[-1])
            tstep = 1
            time_min, time_max = (
              date_min + dt.timedelta(hours=int(h)) for h in (t_min, t_max)
            )
            f.write(
              F"  {i:4} {idx[-1]:3} {idx[-2]:3} "
              F"{t_min:2} {t_max:2}  "
              F"{lon:7.2f} {lat:6.2f} "
              F"{nclev[idx[-2]]:7.2f} "
              F"{time_min:%Y%m%d-%Hh} "
              F"{time_max:%Y%m%d-%Hh} "
              F"{w_min:7.5f} {w_max:7.5f} "
              F"{self.ncdata[t_min, idx[0], idx[1], i]:12.5e} "
              F"{self.ncdata[t_max, idx[0], idx[1], i]:12.5e} "
              F"{vmin:9.2e} {vmax:9.2e} "
              F"\n"
            )
            # print(
            #   F"{self.name:3} ; "
            #   F"lon[{i:04}] = {lon:7.2f} ; "
            #   F"lat[{idx[1]:03}] = {lat:6.2f} ; "
            #   F"lev[{idx[0]:03}] ; "
            #   # F"time[{t_min:02}] = {time:%Y%m%d-%Hh} ; "
            #   F"val[{time_min:%Y%m%d-%Hh}] = "
            #   F"{self.ncdata[t_min, idx[0], idx[1], i]:11.5e}"
            #   F"val[{time_max:%Y%m%d-%Hh}] = "
            #   F"{self.ncdata[t_max, idx[0], idx[1], i]:11.5e}"
            # )


class VarOut(object):
  # -------------------------------------------------------------------
  def __init__(self, name, instru, fileversion):
    # import numpy as np

    variables = {
      "P": {
        "longname": "Surface pressure",
        "outstr": "L2_P_surf_daily_average",
        "mode": "2d",
        "dtype_in": ">f4",
        "dtype_out": np.float32,
        "extralev": 0,
        "ncvars": {
          "sp": VarNC("sp", "2d", 1.e-2, valid_range=(0., 110.e3, )),
        },
        "statfile": None,
      },
      "Q": {
        "longname": "Specific humidity",
        "outstr": "L2_H2O_daily_average",
        "mode": "3d",
        "dtype_in": ">f4",
        "dtype_out": np.float32,
        "extralev": 0,
        "ncvars": {
          "q": VarNC("q", "3d", instru.f_q, valid_range=(0., 50.e-3, )),
        },
        "statfile": None,
      },
      "T": {
        "longname": "Temperature",
        "outstr": "L2_temperature_daily_average",
        "mode": "3d",
        "dtype_in": ">f4",
        "dtype_out": np.float32,
        "extralev": 2,
        "ncvars": {
          "ta": VarNC("ta", "3d", 1., valid_range=(150., 400., )),
          "skt": VarNC("skt", "2d", 1., valid_range=(150., 400., )),
        },
        "statfile": "L2_status",
      },
      "S": {
        "longname": "Surface type",
        "outstr": "L2_SurfType",
        "mode": "2d",
        "dtype_in": ">i4",
        "dtype_out": np.int32,
        "extralev": 0,
        "ncvars": {
          "ci": VarNC("ci", "2d", 1., altname="siconc"),
          "lsm": VarNC("lsm", "2d", 1.),
          "sd": VarNC("sd", "2d", 1.),
        },
        "statfile": None,
      },
    }

    self.name = name
    self.longname = variables[name]["longname"]
    self.outstr = variables[name]["outstr"]
    self.mode = variables[name]["mode"]
    self.dtype_in = variables[name]["dtype_in"]
    self.dtype_out = variables[name]["dtype_out"]
    self.extralev = variables[name]["extralev"]
    self.ncvars = variables[name]["ncvars"]
    self.statfile = variables[name]["statfile"]

    self.fileversion = fileversion
    self.instru = instru.name
    self.ampm = instru.ampm

  # -------------------------------------------------------------------
  def __bool__(self):
    return True

  # -------------------------------------------------------------------
  def __repr__(self):
    return (
      F"{self.longname} ({self.mode})"
    )

  # -------------------------------------------------------------------
  def init_datas(self, ncgrid, tggrid):

    # import numpy as np

    if self.mode == "2d":
      tgshape = (tggrid.nlat, tggrid.nlon)
    else:
      tgshape = (tggrid.nlev + self.extralev, ncgrid.nlat, ncgrid.nlon)

    self.tgprofiles = np.ma.empty(tgshape)
    if self.statfile:
      self.stprofiles = np.ma.empty(tgshape[1:])

    for Vnc in self.ncvars.values():
      Vnc.init_datas(ncgrid)

    # if self.mode == "2d":
    #   ncshape = tgshape = (ncgrid.nlat, ncgrid.nlon)
    # else:
    #   if self.name == "temp":
    #     tg_nlev = tggrid.nlev + 2
    #   else:
    #     tg_nlev = tggrid.nlev
    #   ncshape = (ncgrid.nlev, ncgrid.nlat, ncgrid.nlon)
    #   tgshape = (tg_nlev, ncgrid.nlat, ncgrid.nlon)

    # print(ncshape, tgshape)

    # # self.ncprofiles = np.full(ncshape, np.nan)
    # # self.tgprofiles = np.full(tgshape, np.nan)
    # self.ncprofiles = np.ma.empty(ncshape)
    # self.tgprofiles = np.ma.empty(tgshape)
    # self.ncdata = None

  # -------------------------------------------------------------------
  def clear_datas(self):

    del self.tgprofiles
    if self.statfile:
      del self.stprofiles

    for Vnc in self.ncvars.values():
      Vnc.clear_datas()

  # -------------------------------------------------------------------
  def print_values(self):

    fmt = ".2f"
    print(
      F"{self.longname} ({self.mode}): "
      F"min={self.tgprofiles.min():{fmt}} ; "
      F"max={self.tgprofiles.max():{fmt}} ; "
      F"mean={self.tgprofiles.mean():{fmt}} ; "
      F"std={self.tgprofiles.std():{fmt}}"
      # F"min={values.min():{fmt}} {self.units} ; "
      # F"max={values.max():{fmt}} {self.units} ; "
      # F"mean={values.mean():{fmt}} {self.units} ; "
      # F"std={values.std():{fmt}} {self.units}"
    )

  # -------------------------------------------------------------------
  def fileout(self, date, ftype="data"):

    if ftype == "data":
      string = self.outstr
    elif ftype == "status":
      string = self.statfile

    if string:
      ret = (
        F"unmasked_ERA5_{self.instru}_{string}."
        F"{date:%Y%m%d}."
        F"{self.ampm}_{self.fileversion}"
      )
    else:
      ret = None

    return ret

  # -------------------------------------------------------------------
  def dirout(self, dirout, date):

    if self.outstr:
      ret = dirout.joinpath(F"{date:%Y}", F"{date:%m}")
    else:
      ret = None

    return ret

  # -------------------------------------------------------------------
  def pathout(self, dirout, date, ftype="data"):

    if ftype == "data":
      string = self.outstr
    elif ftype == "status":
      string = self.statfile

    if string:
      ret = self.dirout(dirout, date).joinpath(self.fileout(date, ftype))
    else:
      ret = None

    return ret


class Variable(object):
  # -------------------------------------------------------------------
  def __init__(self, name, instru, fileversion):

    # import numpy as np

    self.name = name

    if name == "Psurf":
      self.longname = "Surface pressure"
      self.ncvar = "sp"
      self.ncvar_alt = None
      self.mode = "2d"
      self.coeff = 1.e-2
      self.str = "L2_P_surf_daily_average"
      self.units = "hPa"
    if name == "Tsurf":
      self.longname = "Skin temperature"
      self.ncvar = "skt"
      self.ncvar_alt = None
      self.mode = "2d"
      self.coeff = 1.
      self.str = None
      self.units = "K"
    if name == "temp":
      self.longname = "Temperature"
      self.ncvar = "ta"
      self.ncvar_alt = None
      self.mode = "3d"
      self.coeff = 1.
      self.str = "L2_temperature_daily_average"
      self.units = "K"
    if name == "h2o":
      self.longname = "Specific humidity"
      self.ncvar = "q"
      self.ncvar_alt = None
      self.mode = "3d"
      self.coeff = instru.f_q
      self.str = "L2_H2O_daily_average"
      self.units = F"{self.coeff ** -1:1.0e} kg kg**-1"

    if name == "ci":
      self.longname = "Sea-ice cover"
      self.ncvar = "ci"
      self.ncvar_alt = "siconc"  # new var name since 01/2018
      self.mode = "2d"
      self.coeff = 1.
      self.str = None
      self.units = "(0-1)"
    if name == "sd":
      self.longname = "Snow depth"
      self.ncvar = "sd"
      self.ncvar_alt = None
      self.mode = "2d"
      self.coeff = 1.
      self.str = None
      self.units = "m of water equiv."
    if name == "lsm":
      self.longname = "Land-sea mask"
      self.ncvar = "lsm"
      self.ncvar_alt = None
      self.mode = "2d"
      self.coeff = 1.
      self.str = None
      self.units = "(0-1)"
    if name == "surftype":
      self.longname = "Surface type"
      self.ncvar = "lsm"
      # self.ncvar = None
      self.ncvar_alt = None
      self.mode = "2d"
      self.coeff = 1.
      self.str = "L2_SurfType"
      self.units = "[1, 2, 3]"

    if name == "stat":
      self.longname = "Temp status"
      self.ncvar = None
      self.ncvar_alt = None
      self.mode = "2d"
      self.coeff = None
      self.str = "L2_status"

    if name == "time":
      self.longname = "Time"
      self.ncvar = "time"
      self.ncvar_alt = None
      self.mode = "2d"
      self.coeff = 1.
      self.str = None

    self.fileversion = fileversion
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

    # import numpy as np

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

    # self.ncprofiles = np.full(ncshape, np.nan)
    # self.tgprofiles = np.full(tgshape, np.nan)
    self.ncprofiles = np.ma.empty(ncshape)
    self.tgprofiles = np.ma.empty(tgshape)
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

    # import numpy as np
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

    # import numpy as np

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
  # import numpy as np

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
