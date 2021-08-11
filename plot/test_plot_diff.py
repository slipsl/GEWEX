#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This must come first
from __future__ import print_function, unicode_literals, division

# Standard library imports
# ========================
from pathlib import Path
import os.path
import datetime as dt
import numpy as np
from scipy.io import FortranFile
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import pprint
pp = pprint.PrettyPrinter(indent=2)


#######################################################################

def cm2inch(x):

  return x / 2.54


#######################################################################

fig, (ax1, ax2, ax3) = plt.subplots(
  # figsize=(15, 10), dpi=300
  figsize=(cm2inch(21.0), cm2inch(29.7)),
  nrows=3, ncols=1,
  sharex="all",
  # sharey="all",
  # dpi=300,
)

lat = np.array([i / 4. for i in range(-90*4, 90*4+1)])
lon = np.array([i / 4. for i in range(-180*4, 180*4)])
nlat = len(lat)
nlon = len(lon)

cond = lon < 0.
lon[cond] = lon[cond] + 360.  # 0. <= lon < 360.

dirnc = Path(os.path.join("input", "AN_SF", "2008"))
filenc = dirnc.joinpath(
  "sp.200802.as1e5.GLOBAL_025.nc"
)

fileout = "unmasked_ERA5_AIRS_V6_L2_P_surf_daily_average.20080220.AM_05"

print("Read netcdf")
with Dataset(filenc, "r", format="NETCDF4") as f_nc:
  nc_lat = f_nc.variables["latitude"][:]
  nc_lon = f_nc.variables["longitude"][:]
  Pnc = f_nc.variables["sp"][:]
Pnc = Pnc[24*19, :, :] / 100.

print(nc_lat[0], nc_lat[360], nc_lat[720])
print(lat[0], lat[360], lat[720])
print(nc_lon[0], nc_lon[720], nc_lon[1439])
print(lon[0], lon[720], lon[1439])

print(
  np.where(nc_lat == 0),
  np.where(nc_lon == 0),
  np.where(lat == 0),
  np.where(lon == 0),
)


print(
  Pnc[np.where(nc_lat == 0), :],
)

print(np.roll(lon, 720))

exit()

sorted_lat_idx = nc_lat.argsort()
sorted_lon_idx = nc_lon.argsort()

Pnc = Pnc[sorted_lat_idx, :]
Pnc = Pnc[:, sorted_lon_idx]

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
im1 = ax1.scatter(x=X, y=Y, c=Z, cmap="RdGy")
fig.colorbar(im1, ax=ax1)
ax1.set_title("P_ori - P_new")


print("Prepare data")
X = np.empty(Pnc.size)
Y = np.empty(Pnc.size)
Z = np.empty(Pnc.size)
it = np.nditer(Pnc, flags=["multi_index"])

for i, x in enumerate(it):
  # print(x, Pnew[it.multi_index], Pold[it.multi_index])
  ilon = it.multi_index[1]
  ilat = it.multi_index[0]
  X[i] = nc_lon[ilon]
  Y[i] = lat[ilat]
  Z[i] = x - Pnew[it.multi_index]

print(Z.min(), Z.max())

print("Plot data")
# ax2.subplot(312)
im2 = ax2.scatter(x=X, y=Y, c=Z, cmap="RdGy")
fig.colorbar(im2, ax=ax2)
ax2.set_title("P_netcdf")


print("Prepare data")
X = np.empty(Pnc.size)
Y = np.empty(Pnc.size)
Z = np.empty(Pnc.size)
it = np.nditer(Pnew, flags=["multi_index"])

for i, x in enumerate(it):
  # print(x, Pnew[it.multi_index], Pold[it.multi_index])
  ilon = it.multi_index[1]
  ilat = it.multi_index[0]
  X[i] = lon[ilon]
  Y[i] = lat[ilat]
  Z[i] = x  # - Pnew[it.multi_index]

print(Z.min(), Z.max())

print("Plot data")
# ax2.subplot(312)
im3 = ax3.scatter(x=X, y=Y, c=Z, cmap="RdGy")
fig.colorbar(im3, ax=ax3)
ax3.set_title("P_new")




print("Save fig")

ax1.set_xlim([0., 360.])
ax1.set_ylim([-90., 90.])
ax1.set_xticks(range(0, 361, 30))
ax1.set_yticks(range(-90, 91, 30))

plt.setp(
  (ax1, ax2, ax3),
  xlim=[0., 360.],
  ylim=[-90., 90.],
  xticks=range(0, 361, 30),
  yticks=range(-90, 91, 30),
  # xticklabels=['a', 'b', 'c'],
)


fig.text(
  0.95, 0.05,
  # "test",
  F"{dt.datetime.now():%d/%m/%Y %H:%M:%S}",
  fontsize=8,
  fontstyle="italic",
  ha="right",
)
# fig.tight_layout()
fig.savefig("Pdiff.png")
# plt.show()
