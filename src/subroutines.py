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
import os
import datetime as dt
# from cftime import num2date, date2num
import pprint


import numpy as np
import pandas as pd
from fortio import FortranFile
from netCDF4 import Dataset

pp = pprint.PrettyPrinter(indent=2)


#######################################################################
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


#----------------------------------------------------------------------
def get_fileout(varname, date_curr):

  return (
    F"unmasked_ERA5_{instrument}_{varname}.{date_curr:%Y%m%d}.{ampm}_{fileversion}"
  )


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
    if not os.path.isfile(filename):
      print(F"Input file missing: {filename}")
      return None

  univT = [i - 24. for i in range(72)]

  off_prev = (date_prev.day - 1) * 24
  off_curr = (date_curr.day - 1) * 24
  off_next = (date_next.day - 1) * 24

  # Longitudes => timesteps
  print(F"Reading {file_curr} to process t = f(lon)\n"+80*"=")
  with Dataset(filename, "r", format="NETCDF4") as f_in:
    nc_lat  = f_in.variables["latitude"]
    nc_lon  = f_in.variables["longitude"]
    nc_time = f_in.variables["time"]

    nlat = nc_lat.size
    nlon = nc_lon.size

  print(type(nc_lon))

  exit()
