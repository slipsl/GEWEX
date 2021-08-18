#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This must come first
from __future__ import print_function, unicode_literals, division

# Standard library imports
# ========================
import sys
from pathlib import Path
import os.path
from math import floor, ceil
import datetime as dt
import numpy as np
from scipy.io import FortranFile
# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as clr
# from mpl_toolkits.basemap import Basemap
# from netCDF4 import Dataset
import pprint

pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================
sys.path.append(str(Path(__file__).resolve().parents[1].joinpath("src")))
import gewex_param as gw
import gewex_netcdf as gnc


#######################################################################
def get_arguments():
  from argparse import ArgumentParser
  from argparse import RawTextHelpFormatter

  parser = ArgumentParser(
    formatter_class=RawTextHelpFormatter
  )

  parser.add_argument(
    "mode", action="store",
    # type=int,
    choices=["plot", "diff"],
    help=(
      "Either plot the variable itself or "
      "diffences between ref and variable"
    )
  )
  parser.add_argument(
    "varname", action="store",
    # type=int,
    choices=["Psurf", "Tsurf", "temp", "h2o"],
    help=(
      "Variable to plot: \"Psurf\", \"Tsurf\", \"temp\", \"h2o\""
    )
  )
  parser.add_argument(
    "runtype", action="store",
    type=int,
    choices=[1, 2, 3, 4, 5],
    help=(
      "Run type:\n"
      "  - 1 = AIRS / AM\n"
      "  - 2 = AIRS / PM\n"
      "  - 3 = IASI / AM\n"
      "  - 4 = IASI / PM\n"
      "  - 5 = Test mode (node = 0.0)\n"
    )
  )
  parser.add_argument(
    "date", action="store",
    type=lambda s: dt.datetime.strptime(s, '%Y%m%d'),
    help="Date: YYYYMMJJ"
  )
  parser.add_argument(
    "-c", "--contour", action="store_true",
    help="Use contour instead of scatter plot"
  )
  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )

  return parser.parse_args()


#----------------------------------------------------------------------
def cm2inch(x):

  return x / 2.54


#----------------------------------------------------------------------
def read_f77(filein, grid):

  with FortranFile(filein, "r", header_dtype=">u4") as f:
    rec = f.read_record(dtype=">f4")
  var_out = rec.reshape(grid.nlon, grid.nlat).T

  return var_out


# #----------------------------------------------------------------------
# def read_netcdf(filein, varname):

#   with Dataset(filein, "r", format="NETCDF4") as f:
#     nc_lat = f.variables["latitude"][:]
#     nc_lon = f.variables["longitude"][:]
#     var_out = f.variables[varname][:]
#   var_out = var_out[24*19, :, :] / 100.

#   # print("Adapt variable to lat/lon grid")
#   var_out = np.flipud(var_out)
#   idx_lon = np.where(nc_lon == 180.)[0][0]
#   var_out = np.roll(var_out, idx_lon, axis=1)

#   return var_out


#----------------------------------------------------------------------
def prep_data(var_in, grid, fg_cntr=True):

  if fg_cntr:
    X = grid.lon
    Y = grid.lat
    Z = var_in
  else:
    X = np.tile(grid.lon, grid.lat.size)
    Y = np.repeat(grid.lat, grid.lon.size)
    Z = var_in.flatten()
    # X = np.empty(var_in.size)
    # Y = np.empty(var_in.size)
    # Z = np.empty(var_in.size)
    # it = np.nditer(var_in, flags=["multi_index"])
    # for i, x in enumerate(it):
    #   ilon = it.multi_index[1]
    #   ilat = it.multi_index[0]
    #   X[i] = grid.lon[ilon]
    #   Y[i] = grid.lat[ilat]
    #   Z[i] = x

  return (X, Y, Z)


#----------------------------------------------------------------------
def plot_data(fg_cntr, ax, X, Y, Z, cmap, norm, title, nb=30):

  if fg_cntr:
    im = contour_plot(ax, X, Y, Z, cmap, title, nb=nb)
  else:
    im = scatter_plot(ax, X, Y, Z, cmap, norm, title)
  # im = ax.scatter(x=X, y=Y, c=Z, cmap=cmap, norm=norm)
  cb = fig.colorbar(im, ax=ax)
  ax.set_title(title, loc="left")
  config_axcb(ax, cb, X, Y)
  write_comment(ax, Z)


#----------------------------------------------------------------------
def scatter_plot(ax, X, Y, Z, cmap, norm, title):

  im = ax.scatter(x=X, y=Y, c=Z, cmap=cmap, norm=norm)
  # cb = fig.colorbar(im, ax=ax)
  # ax.set_title(title, loc="left")
  # cb = plot(ax, im, title)
  # config_axcb(ax, cb, X, Y)

  return im


#----------------------------------------------------------------------
def contour_plot(ax, X, Y, Z, cmap, title, nb=30):

  im = ax.contourf(X, Y, Z, levels=nb, cmap=cmap)
  # cb = fig.colorbar(im, ax=ax)
  # ax.set_title(title, loc="left")
  # plot(ax, cb, im, title)
  # config_axcb(ax, cb, X, Y)

  return im


#----------------------------------------------------------------------
def plot(ax, im, title):

  cb = fig.colorbar(im, ax=ax)
  ax.set_title(title, loc="left")

  return cb


#----------------------------------------------------------------------
def write_comment(ax, Z):

  ax.text(
    0.98, 0.02,
    F"m = {Z.mean():.2e} / $\sigma$ = {Z.std():.2e}",
    fontsize=8,
    ha="right",
    va="center",
    transform=ax.transAxes,
  )
  ax.text(
    0.98, 0.96,
    # F"min = {Z.min():.2e} / max = {Z.max():.2e}",
    F"{Z.min():.2e} < $\Delta$ < {Z.max():.2e}",
    fontsize=8,
    ha="right",
    va="center",
    transform=ax.transAxes,
  )


#----------------------------------------------------------------------
def config_axcb(ax, cb, X, Y):

  # print(X.min(), X.max())
  # print(Y.min(), Y.max())
  # grid = tg_grid
  (ymin, xmin) = (floor(l) for l in (Y.min(), X.min()))
  (ymax, xmax) = (ceil(l)  for l in (Y.max(), X.max()))

  cb.ax.tick_params(labelsize=8)
  ax.tick_params(axis='both', which='major', labelsize=8)
  ax.tick_params(axis='x', labelrotation=45)

  ax.set_xticks(range(xmin, xmax + 1, 30))
  ax.set_yticks(range(ymin, ymax + 1, 30))

  ax.set_xlim([float(xmin), float(xmin) + 360.])
  ax.set_ylim([float(ymin), float(ymax)])



#######################################################################

if __name__ == "__main__":

  # .. Initialization ..
  # ====================
  # ... Command line arguments ...
  # ------------------------------
  args = get_arguments()
  if args.verbose:
    print(args)

  project_dir = Path(__file__).resolve().parents[1]
  dir_img = project_dir.joinpath("img")
  dir_idl = project_dir.joinpath("output", "exemples")


  cmap_plot = "coolwarm"
  cmap_diff = "seismic"
  # cmap_diff = "RdGy"

  if args.mode == "plot":
    cmap = cmap_plot
  else:
    cmap = cmap_diff

  instru = gw.InstruParam(args.runtype)
  params = gw.GewexParam(project_dir)

  print(instru)
  print(params)

  variable = gw.Variable(args.varname, instru)
  print(variable)

  if variable.mode == "3d":
    print("3D variable not implemented yet")
    # exit()

  # print(variable.get_ncfiles(params.dirin, args.date))
  # print(variable.pathout(params.dirout, args.date))
  # print(variable.dirout(params.dirout, args.date))
  # print(variable.fileout(args.date))

  img_type = "png"
  img_name = (
    F"{args.varname}_{args.date:%Y%M%d}_"
    F"{instru.name}_{instru.ampm}_"
    F"{args.mode}"
  )
  print(img_name)


  fig_title = (
    F"{variable.longname} ({variable.units})\n"
    F"{args.date:%d/%m/%Y} "
    F"{instru.name}_{instru.ampm}"
  )

  variable.ncfiles = variable.get_ncfiles(params.dirin, args.date)
  variable.pyfile = variable.pathout(params.dirout, args.date)
  variable.idlfile = dir_idl.joinpath(variable.fileout(args.date))
  pp.pprint(
    tuple(str(f) for f in (variable.ncfiles, variable.pyfile, variable.idlfile))
  )

  if not variable.idlfile.exists():
    fg_idl = False
  else:
    fg_idl = True

  # ... Load NetCDF & target grids ...
  # ----------------------------------
  nc_grid = gnc.NCGrid()
  nc_grid.load(variable.ncfiles)
  tg_grid = gw.TGGrid()
  tg_grid.load(nc_grid)

  A4R = (cm2inch(21.0), cm2inch(29.7))
  A4L = A4R[::-1]

  print(A4R, A4L)

  if variable.mode == "2d":
    figsize = A4R
    nrows = 3
    ncols = 1
  else:
    figsize = A4L
    nrows = 3
    ncols = 3


  fig, axes = plt.subplots(
    figsize=figsize,
    nrows=nrows, ncols=ncols,
    # sharex="all",
    # sharey="all",
    # dpi=300,
  )
  # cbars = {}
  # imags = {}



  print("Read input files")
  # ========================

  print(" - netcdf")
  # ------------------------
  nstep = 24  # number of time steps per day
  i_time = (args.date.day - 1) * nstep
  variable.ncvalues = (  # NetCDF grid
      gnc.read_netcdf(variable, nc_grid, slice(nc_grid.nlon), i_time)
  )
  variable.ncvalues = (  # F77 grid
      gw.grid_nc2tg(variable.ncvalues, nc_grid, tg_grid)
  )
  print(variable.ncvalues.shape)

  print(" - python")
  # ------------------------
  variable.pyvalues = \
      read_f77(variable.pyfile, tg_grid)
  print(variable.pyvalues.shape)

  if fg_idl:
    print(" - idl")
    # ------------------------
    variable.idlvalues = \
        read_f77(variable.idlfile, tg_grid)
    print(variable.idlvalues.shape)


  print("Prepare data")
  # ========================
  if args.mode == "plot":
    print(" - netcdf")
    # ------------------------
    (X0, Y0, Z0) = prep_data(variable.ncvalues, tg_grid, args.contour)
    print(Z0.mean(), Z0.std())
    norm0 = None
    title0 = "netcdf"
    print(" - python")
    # ------------------------
    (X1, Y1, Z1) = prep_data(variable.pyvalues, tg_grid, args.contour)
    print(Z1.mean(), Z1.std())
    norm1 = None
    title1 = "python"
    if fg_idl:
      print(" - idl")
      # ------------------------
      (X2, Y2, Z2) = prep_data(variable.idlvalues, tg_grid, args.contour)
      print(Z2.mean(), Z2.std())
      norm2 = None
      title2 = "idl"
  else:
    print(" - (netcdf - python)")
    # ------------------------
    (X0, Y0, Z0) = prep_data(
      variable.ncvalues - variable.pyvalues, tg_grid, args.contour
    )
    print(Z0.mean(), Z0.std())
    norm0 = clr.TwoSlopeNorm(vcenter=0., vmin=Z0.min(), vmax=Z0.max())
    title0 = "nc - py"
    if fg_idl:
      print(" - (netcdf - idl)")
      # ------------------------
      (X1, Y1, Z1) = prep_data(
        variable.ncvalues - variable.idlvalues, tg_grid, args.contour
      )
      print(Z1.mean(), Z1.std())
      norm1 = clr.TwoSlopeNorm(vcenter=0., vmin=Z1.min(), vmax=Z1.max())
      title1 = "nc - idl"
      print(" - (idl - python)")
      # ------------------------
      (X2, Y2, Z2) = prep_data(
        variable.idlvalues - variable.pyvalues, tg_grid, args.contour
      )
      print(Z2.mean(), Z2.std())
      norm2 = clr.TwoSlopeNorm(vcenter=0., vmin=Z2.min(), vmax=Z2.max())
      title2 = "idl - py"

  print("Plot data")
  # ========================
  plot_data(args.contour, axes[0], X0, Y0, Z0, cmap, norm0, title0)
  plot_data(args.contour, axes[1], X1, Y1, Z1, cmap, norm1, title1)
  plot_data(args.contour, axes[2], X2, Y2, Z2, cmap, norm2, title2)
  # scatter_plot(axes[0], X0, Y0, Z0, cmap, norm0, title0)
  # scatter_plot(axes[1], X1, Y1, Z1, cmap, norm1, title1)
  # scatter_plot(axes[2], X2, Y2, Z2, cmap, norm2, title2)


  print("Plot config")
  # ========================
  plt.suptitle(fig_title)
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

  # grid = tg_grid
  # (lat_min, lon_min) = (floor(l) for l in (grid.lat.min(), grid.lon.min()))
  # (lat_max, lon_max) = (ceil(l)  for l in (grid.lat.max(), grid.lon.max()))
  # # lat_min = floor(grid.lat.min())
  # # lat_max = ceil(grid.lat.max())
  # # lon_min = floor(grid.lon.min())
  # # lon_max = ceil(grid.lon.max())
  # for ax in axes:
  #   ax.tick_params(axis='both', which='major', labelsize=8)
  #   ax.tick_params(axis='x', labelrotation=45)
  #   # ax.tick_params(axis='both', which='minor', labelsize=8)

  #   ax.set_xlim([float(lon_min), float(lon_min) + 360.])
  #   ax.set_ylim([float(lat_min), float(lat_max)])
  #   ax.set_xticks(range(lon_min, lon_max + 1, 30))
  #   ax.set_yticks(range(lat_min, lat_max + 1, 30))
  #   # ax.set_xlim([grid.lon.min(), grid.lon.min() + 360.])
  #   # ax.set_ylim([grid.lat.min(), grid.lat.max()])
  #   # ax.set_xticks(
  #   #   range(floor(grid.lon.min()), ceil(grid.lon.max())+1, 30)
  #   # )
  #   # ax.set_yticks(
  #   #   range(floor(grid.lat.min()), ceil(grid.lat.max())+1, 30)
  #   # )
  # for cb in cbars.values():
  #   cb.ax.tick_params(labelsize=8)

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
    dir_img.joinpath(F"{img_name}.{img_type}"),
    # bbox_inches="tight",
  )


  # plt.show()


  exit()



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

fileout = "unmasked_ERA5_AIRS_V6_L2_P_surf_daily_average.20080220.PM_05"
dirold = Path(os.path.join("output", "exemples"))
fileold = dirold.joinpath(fileout)
dirnew = Path(os.path.join("output", "2008", "02"))
filenew = dirnew.joinpath(fileout)

# fileold = project_dir.joinpath("Ptest.dat")


print("Read netcdf")
# ========================
Pnc = read_netcdf(filenc, "sp")

print("Prepare data")
# ========================
(X, Y, Z) = prep_data(Pnc)
print(
  Z.mean(), Z.std()
)

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
  ax.tick_params(axis='x', labelrotation=45)
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
