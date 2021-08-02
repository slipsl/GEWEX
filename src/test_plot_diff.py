#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This must come first
from __future__ import print_function, unicode_literals, division

# Standard library imports
# ========================
from pathlib import Path
import os.path
import numpy as np
from scipy.io import FortranFile
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import pprint
pp = pprint.PrettyPrinter(indent=2)


#######################################################################

fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

lat = np.array([i / 4. for i in range(-90*4, 90*4+1)])
lon = np.array([i / 4. for i in range(-180*4, 180*4)])
nlat = len(lat)
nlon = len(lon)

cond = lon < 0.
lon[cond] = lon[cond] + 360.

fileout = "unmasked_ERA5_AIRS_V6_L2_P_surf_daily_average.20080220.AM_05"

dirold = Path(os.path.join("output", "exemples"))
fileold = dirold.joinpath(fileout)

dirnew = Path(os.path.join("output", "2008", "02"))
filenew = dirnew.joinpath(fileout)

print("Read original output file")
with FortranFile(fileold, "r", header_dtype=">u4") as f_old:
  rec = f_old.read_record(dtype=">f4")
  Pold = rec.reshape(nlon, nlat).T
print("Read new output file")
with FortranFile(filenew, "r", header_dtype=">u4") as f_new:
  rec = f_new.read_record(dtype=">f4")
  Pnew = rec.reshape(nlon, nlat).T

print("Prepare data")
X = np.empty(Pold.size)
Y = np.empty(Pold.size)
Z = np.empty(Pold.size)
it = np.nditer(Pold, flags=["multi_index"])

for i, x in enumerate(it):
  # print(x, Pnew[it.multi_index], Pold[it.multi_index])
  ilon = it.multi_index[1]
  ilat = it.multi_index[0]
  X[i] = lon[ilon]
  Y[i] = lat[ilat]
  Z[i] = x - Pnew[it.multi_index]

print(Z.min(), Z.max())

print("Plot data")
# ax2.subplot(312)
im = ax.scatter(x=X, y=Y, c=Z, cmap="RdGy")
fig.colorbar(im, ax=ax)
ax.set_title("P (ori) - P  (new)")

print("Save fig")
fig.savefig("Pdiff.png")
# plt.show()
