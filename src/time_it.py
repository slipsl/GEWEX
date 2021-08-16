#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard library imports
# ========================
import timeit

#######################################################################

setup_code = """
import numpy as np
from netCDF4 import Dataset
# filenc = "input/AN_SF/2008/sp.200801.as1e5.GLOBAL_025.nc"
# ncvar = "sp"
filenc = "input/AN_PL/2008/ta.200801.ap1e5.GLOBAL_025.nc"
ncvar = "ta"
lon = 720
t1 = 435
t2 = 436
"""

main_code_1 = """
with Dataset(filenc, "r", format="NETCDF4") as f_in:
  # var = f_in.variables[ncvar][(t1, t2), :, lon].copy()
  var = f_in.variables[ncvar][(t1, t2), :, :, lon].copy()
"""

main_code_2 = """
with Dataset(filenc, "r", format="NETCDF4") as f_in:
  # var = f_in.variables[ncvar][t1:t2+1, :, lon].copy()
  var = f_in.variables[ncvar][t1:t2+1, :, :, lon].copy()
"""

main_code_3 = """
with Dataset(filenc, "r", format="NETCDF4") as f_in:
  # var1 = f_in.variables[ncvar][(t1, ), :, lon].copy()
  var1 = f_in.variables[ncvar][(t1, ), :, :, lon].copy()
with Dataset(filenc, "r", format="NETCDF4") as f_in:
  # var2 = f_in.variables[ncvar][(t2, ), :, lon].copy()
  var2 = f_in.variables[ncvar][(t2, ), :, :, lon].copy()
var = np.ma.concatenate([var1, var2], axis=0)
"""

main_code_4 = """
vars = []
with Dataset(filenc, "r", format="NETCDF4") as f_in:
  for lev in range(37):
    vars.append(
      f_in.variables[ncvar][(t1, t2), (lev, ), :, lon].copy()
    )
var = np.ma.concatenate(vars, axis=1)
"""

number = 250

time1 = timeit.timeit(
  stmt=main_code_1,
  setup=setup_code,
  number=number
)
print(
  F"Lecture x1 (indx) ; 2 ts\n  "
  F"{time1} ({time1/number})"
)

time2 = timeit.timeit(
  stmt=main_code_2,
  setup=setup_code,
  number=number
)
print(
  F"Lecture x1 (slice) + concatenate\n  "
  F"{time2} ({time2/number})"
)

time3 = timeit.timeit(
  stmt=main_code_3,
  setup=setup_code,
  number=number
)
print(
  F"Lecture x2 + concatenate\n  "
  F"{time3} ({time3/number})"
)

time4 = timeit.timeit(
  stmt=main_code_4,
  setup=setup_code,
  number=number
)
print(
  F"Lecture (indx) x n_lev + concatenate\n  "
  F"{time4} ({time4/number})"
)
