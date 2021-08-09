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
def read_f77(filein, ilev=None):

  with FortranFile(filein, "r", header_dtype=">u4") as f:
    rec = f.read_record(dtype=">f4")

  if ilev:
    var_out = rec.reshape(nlev, nlon, nlat)
    var_out = np.rollaxis(var_out, 2, 1)
    var_out = var_out[ilev, :, :]
  else:
    var_out = rec.reshape(nlon, nlat).T

  return var_out


#----------------------------------------------------------------------
def read_netcdf(filein, varname, ilev):

  with Dataset(filein, "r", format="NETCDF4") as f:
    nc_lat = f.variables["latitude"][:]
    nc_lon = f.variables["longitude"][:]
    var_out = f.variables[varname][24*19, ilev, :, :] * coeff_h2o

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


#----------------------------------------------------------------------
def write_comment(ax, Z):

  ax.text(
    0.98, 0.02,
    F"m = {Z.mean():5.2f} / $\sigma$ = {Z.std():5.2f}",
    fontsize=8,
    ha="right",
    va="center",
    transform=ax.transAxes,
  )
  ax.text(
    0.98, 0.97,
    # F"min = {Z.min():5.2f} / max = {Z.max():5.2f}",
    F"{Z.min():5.2f} < $\Delta$ < {Z.max():5.2f}",
    fontsize=8,
    ha="right",
    va="center",
    transform=ax.transAxes,
  )


#######################################################################

project_dir = Path(__file__).resolve().parents[1]
dir_img = project_dir.joinpath("img")

img_name = "H2O_Surf"

cmap_plot = "coolwarm"
cmap_diff = "seismic"

coeff_h2o = 1000.

# fig, ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = \
#   plt.subplots(
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, ncols=1,
    # sharex="all",
    sharey="all",
    figsize=(cm2inch(21.0), cm2inch(29.7)),
    # figsize=(15, 10),
    dpi=300,
  )

lat = np.array([i / 4. for i in range(-90*4, 90*4+1)])
lon = np.array([i / 4. for i in range(-180*4, 180*4)])
nlat = len(lat)
nlon = len(lon)
lev = np.array([
    69.71,  86.07,
   106.27, 131.20,
   161.99, 200.00,
   222.65, 247.87,
   275.95, 307.20,
   341.99, 380.73,
   423.85, 471.86,
   525.00, 584.80,
   651.04, 724.78,
   800.00, 848.69,
   900.33, 955.12,
  1013.00,
])
nlev = len(lev)

pl = {
   200: {"nc": 14, "f77": 5},
   650: {"nc": 24, "f77": 16},
   800: {"nc": 28, "f77": 18},
   900: {"nc": 32, "f77": 20},
  1000: {"nc": 36, "f77": 22},
}

# cond = lon < 0.
# lon[cond] = lon[cond] + 360.  # 0. <= lon < 360.

# dirnc = Path(os.path.join("input", "AN_PL", "2008"))
# filenc = dirnc.joinpath(
#   "q.200802.ap1e5.GLOBAL_025.nc"
# )

file_press = "unmasked_ERA5_AIRS_V6_L2_P_surf_daily_average.20080220.AM_05"
fileout = "unmasked_ERA5_AIRS_V6_L2_H2O_daily_average.20080220.AM_05"
dirold = Path(os.path.join("output", "exemples"))
fileold = dirold.joinpath(fileout)
dirnew = Path(os.path.join("output", "2008", "02"))
filenew = dirnew.joinpath(fileout)
fileref = dirnew.joinpath(file_press)


print(F"Read P_surf {fileref}")
# ========================
Pref = read_f77(fileref)
print(F"Read 3D var old {fileold}")
# ========================
Qold = read_f77(fileold, pl[1000]["f77"])
print(F"Read 3D var new {filenew}")
# ========================
Qnew = read_f77(filenew, pl[1000]["f77"])

print("Prepare data old-new")
# ========================
(X, Y, Z) = prep_data(Qold - Qnew)
print(
  Z.mean(), Z.std()
)

cond = abs(Z) > 0.1
pp.pprint(
  X[cond],
)
pp.pprint(
  Y[cond],
)
pp.pprint(
  Z[cond],
)

print("Plot data")
# ========================
im1 = ax1.scatter(x=X, y=Y, c=Z, cmap=cmap_diff, marker=".")
cb1 = fig.colorbar(im1, ax=ax1)
ax1.set_title("old - new\n(1013hPa)")
# ax1.contour(
#   lon, lat, Pref,
#   levels=[1000.00, 1013.00, 1020.00],
#   # levels=[1013.00, 1024.00],
#   linestyles=["dotted", "dashed", "solid"],
#   # levels=[200.00, 651.04, 1013.00],
#   # linestyles=["dotted", "dashed", "solid"],
#   colors="black",
#   linewidths=0.2,
# )
write_comment(ax1, Z)

im2 = ax2.contourf(lon, lat, Pref, 12, cmap=cmap_plot)
cb2 = fig.colorbar(im2, ax=ax2)
ax2.contour(
  lon, lat, Pref,
  levels=[1013.00, 1024.00],
  linestyles=["dotted", "solid"],
  # levels=[200.00, 651.04, 1013.00],
  # linestyles=["dotted", "dashed", "solid"],
  colors="black",
  linewidths=0.2,
)

im3 = ax3.contour(
  lon, lat, Pref,
  levels=[1013.00, 1024.00],
  linestyles=["solid"],
  # levels=[200.00, 651.04, 1013.00],
  # linestyles=["dotted", "dashed", "solid"],
  colors="black",
  linewidths=0.2,
)




# print(" - netcdf")
# Pnc = read_netcdf(filenc, "q", pl[200]["nc"])
# print(" - original output file")
# Pold = read_f77(fileold, pl[200]["f77"])
# print(" - new output file")
# Pnew = read_f77(filenew, pl[200]["f77"])

# print("Prepare data old-new")
# # ========================
# (X, Y, Z) = prep_data(Pold - Pnew)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im1 = ax1.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb1 = fig.colorbar(im1, ax=ax1)
# ax1.set_title("old - new\n(200hPa)")
# write_comment(ax1, Z)

# print("Prepare data nc-old")
# # ========================
# (X, Y, Z) = prep_data(Pnc - Pold)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im2 = ax2.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb2 = fig.colorbar(im2, ax=ax2)
# ax2.set_title("nc - old\n(200hPa)")
# write_comment(ax2, Z)

# print("Prepare data nc-new")
# # ========================
# (X, Y, Z) = prep_data(Pnc - Pnew)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im3 = ax3.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb3 = fig.colorbar(im3, ax=ax3)
# ax3.set_title("nc - new\n(200hPa)")
# write_comment(ax3, Z)


# print(F"Read 650hPa\n{80*'='}")
# # ========================
# print(" - netcdf")
# Pnc = read_netcdf(filenc, "q", pl[650]["nc"])
# print(" - original output file")
# Pold = read_f77(fileold, pl[650]["f77"])
# print(" - new output file")
# Pnew = read_f77(filenew, pl[650]["f77"])

# print("Prepare data old-new")
# # ========================
# (X, Y, Z) = prep_data(Pold - Pnew)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im4 = ax4.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb4 = fig.colorbar(im4, ax=ax4)
# ax4.set_title("(650hPa)")
# write_comment(ax4, Z)

# print("Prepare data nc-old")
# # ========================
# (X, Y, Z) = prep_data(Pnc - Pold)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im5 = ax5.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb5 = fig.colorbar(im5, ax=ax5)
# ax5.set_title("(650hPa)")
# write_comment(ax5, Z)

# print("Prepare data nc-new")
# # ========================
# (X, Y, Z) = prep_data(Pnc - Pnew)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im6 = ax6.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb6 = fig.colorbar(im6, ax=ax6)
# ax6.set_title("(650hPa)")
# write_comment(ax6, Z)


# print(F"Read 1000hPa\n{80*'='}")
# # ========================
# print(" - netcdf")
# Pnc = read_netcdf(filenc, "q", pl[1000]["nc"])
# print(" - original output file")
# Pold = read_f77(fileold, pl[1000]["f77"])
# print(" - new output file")
# Pnew = read_f77(filenew, pl[1000]["f77"])

# print("Prepare data old-new")
# # ========================
# (X, Y, Z) = prep_data(Pold - Pnew)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im7 = ax7.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb7 = fig.colorbar(im7, ax=ax7)
# ax7.set_title("(1000hPa)", fontsize=10,)
# write_comment(ax7, Z)

# print("Prepare data nc-old")
# # ========================
# (X, Y, Z) = prep_data(Pnc - Pold)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im8 = ax8.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb8 = fig.colorbar(im8, ax=ax8)
# ax8.set_title("(1000hPa)")
# write_comment(ax8, Z)

# print("Prepare data nc-new")
# # ========================
# (X, Y, Z) = prep_data(Pnc - Pnew)
# print(
#   Z.mean(), Z.std()
# )

# print("Plot data")
# # ========================
# im9 = ax9.scatter(x=X, y=Y, c=Z, cmap=cmap_diff)
# cb9 = fig.colorbar(im9, ax=ax9)
# ax9.set_title("(1000hPa)")
# write_comment(ax9, Z)


print("Plot config")
# ========================
plt.suptitle(F"Humidité spécifique")
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
for ax in (ax1, ax2, ax3):
  ax.tick_params(axis='both', which='major', labelsize=8)
  ax.tick_params(axis='x', labelrotation = 45)
  # ax.tick_params(axis='both', which='minor', labelsize=8)
  ax.set_xlim([-180., 180.])
  ax.set_ylim([-90., 90.])
  ax.set_xticks(range(-180, 181, 60))
  ax.set_yticks(range(-90, 91, 30))

for cb in (cb1, cb2):
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
