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


#----------------------------------------------------------------------
def read_f77(filein):

  with FortranFile(filein, "r", header_dtype=">u4") as f:
    rec = f.read_record(dtype=">f4")
    var_out = rec.reshape(nlon, nlat).T

  return var_out


#----------------------------------------------------------------------
def read_netcdf(filein, varname):

  with Dataset(filein, "r", format="NETCDF4") as f:
    nc_lat = f.variables["latitude"][:]
    nc_lon = f.variables["longitude"][:]
    var_out = f.variables[varname][:]
  var_out = var_out[24*19, :, :] / 100.

  # print("Adapt variable to lat/lon grid")
  var_out = np.flipud(var_out)
  idx_lon = np.where(nc_lon == 180.)[0][0]
  var_out = np.roll(var_out, idx_lon, axis=1)

  return var_out


#----------------------------------------------------------------------
def prep_data(var_out):

  X = np.empty(var_out.size)
  Y = np.empty(var_out.size)
  Z = np.empty(var_out.size)

  it = np.nditer(var_out, flags=["multi_index"])

  for i, x in enumerate(it):
    ilon = it.multi_index[1]
    ilat = it.multi_index[0]
    X[i] = lon[ilon]
    Y[i] = lat[ilat]
    Z[i] = x

  return (X, Y, Z)



#######################################################################

project_dir = Path(__file__).resolve().parents[1]
dir_img = project_dir.joinpath("img")

img_name = "P_plot"

cmap_plot = "coolwarm"
cmap_diff = "RdGy"

# fig, (ax1, ax2, ax3) = plt.subplots(
fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(
  # figsize=(15, 10), dpi=300
  figsize=(cm2inch(21.0), cm2inch(29.7)),
  nrows=3, ncols=2,
  # sharex="all",
  # sharey="all",
  # dpi=300,
)

lat = np.array([i / 4. for i in range(-90*4, 90*4+1)])
lon = np.array([i / 4. for i in range(-180*4, 180*4)])
nlat = len(lat)
nlon = len(lon)

# cond = lon < 0.
# lon[cond] = lon[cond] + 360.  # 0. <= lon < 360.

dirnc = Path(os.path.join("input", "AN_SF", "2008"))
filenc = dirnc.joinpath(
  "sp.200802.as1e5.GLOBAL_025.nc"
)

fileout = "unmasked_ERA5_AIRS_V6_L2_P_surf_daily_average.20080220.AM_05"
dirold = Path(os.path.join("output", "exemples"))
fileold = dirold.joinpath(fileout)
dirnew = Path(os.path.join("output", "2008", "02"))
filenew = dirnew.joinpath(fileout)



print("Read netcdf")
# ========================
Pnc = read_netcdf(filenc, "sp")

print("Prepare data")
# ========================
(X, Y, Z) = prep_data(Pnc)

print("Plot data")
# ========================
# ax2.subplot(312)
im1 = ax1.scatter(x=X, y=Y, c=Z, cmap=cmap_plot)
cb1 = fig.colorbar(im1, ax=ax1)
im4 = ax4.contourf(lon, lat, Pnc, 12, cmap=cmap_plot)
cb4 = fig.colorbar(im4, ax=ax4)
ax1.set_title("P_netcdf")


print("Read original output file")
# ========================
Pold = read_f77(fileold)

print("Prepare data")
# ========================
(X, Y, Z) = prep_data(Pold)

print("Plot data")
# ========================
# ax2.subplot(312)
im2 = ax2.scatter(x=X, y=Y, c=Z, cmap=cmap_plot)
cb2 = fig.colorbar(im2, ax=ax2)
im5 = ax5.contourf(lon, lat, Pold, 12, cmap=cmap_plot)
cb5 = fig.colorbar(im5, ax=ax5)
ax2.set_title("P_ori")


print("Read new output file")
# ========================
Pnew = read_f77(filenew)

print("Prepare data")
# ========================
(X, Y, Z) = prep_data(Pnew)

print("Plot data")
# ========================
im3 = ax3.scatter(x=X, y=Y, c=Z, cmap=cmap_plot)
cb3 = fig.colorbar(im3, ax=ax3)
im6 = ax6.contourf(lon, lat, Pnew, 12, cmap=cmap_plot)
cb6 = fig.colorbar(im6, ax=ax6)
ax3.set_title("P_new")


print("Plot config")
# ========================
plt.suptitle(F"Pression de surface (hPa)")
now = dt.datetime.now()
fig.text(
  0.98, 0.02,
  # "test",
  F"{now:%d/%m/%Y %H:%M:%S}",
  # F"{dt.datetime.now():%d/%m/%Y %H:%M:%S}",
  fontsize=8,
  fontstyle="italic",
  ha="right",
)

# plt.setp(
#   (ax1, ax2, ax3, ax4, ax5, ax6),
#   # xlim=[0., 360.],
#   # xticks=range(0, 361, 30),
#   xlim=[-180., 180.],
#   xticks=range(-180, 181, 30),
#   ylim=[-90., 90.],
#   yticks=range(-90, 91, 30),
# )

# We change the fontsize of minor ticks label 
for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
  ax.tick_params(axis='both', which='major', labelsize=8)
  ax.tick_params(axis='x', labelrotation = 45)
  # ax.tick_params(axis='both', which='minor', labelsize=8)
  ax.set_xlim([-180., 180.])
  ax.set_ylim([-90., 90.])
  ax.set_xticks(range(-180, 181, 30))
  ax.set_yticks(range(-90, 91, 30))

for cb in (cb1, cb2, cb3, cb4, cb5, cb6):
  cb.ax.tick_params(labelsize=8)

# set the spacing between subplots
# fig.tight_layout()
plt.subplots_adjust(
  left=0.075,    # 0.125    left side of the subplots of the figure
  right=0.950,   # 0.900    right side of the subplots of the figure
  bottom=0.090,  # 0.110    bottom of the subplots of the figure
  top=0.900,     # 0.880    top of the subplots of the figure
  wspace=0.150,  # 0.200    amount of width reserved for space between subplots
  hspace=0.300,  # 0.200    amount of height reserved for space between subplots
)

print(F"Save fig {now:%d/%m/%Y %H:%M:%S}")
# ========================
# plt.show()
fig.savefig(
  # "P_plot.png",
  # dir_img.joinpath(img_name),
  dir_img.joinpath(F"{img_name}.png"),
  # bbox_inches="tight",
)
