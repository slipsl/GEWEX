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
 
    # self.fileversion = "05"

    # 1e-3 m == 0.1 cm of equivalent water == 0.5 cm of snow (18.08.2021)
    self.snowdepth_thresh = 1.e-3

    # print(platform.node())
    ipsl = ("ciclad", "camelot", "loholt1")
    # if "ciclad" in socket.gethostname():
    if any(h in socket.gethostname() for h in ipsl):
      self.dirin = Path("/bdd/ERA5/NETCDF/GLOBAL_025/hourly")
      self.dirout = Path("/data/slipsl/GEWEX/ERA5_averages")
      # self.dirout = Path("/bdd/CIRS-LMD/ERA5_averages")
    else:
      self.dirin = project_dir.joinpath("input")
      self.dirout = project_dir.joinpath("output")


  # -------------------------------------------------------------------
  def __repr__(self):
    return (
      # F"File version: {self.fileversion}\n"
      F"Input dir:    {self.dirin}\n"
      F"Output dir:   {self.dirout}"
    )
