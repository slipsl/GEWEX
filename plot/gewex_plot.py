#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from matplotlib.backends.backend_pdf import PdfPages
# from mpl_toolkits.basemap import Basemap
# from netCDF4 import Dataset
import pprint

pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================
sys.path.append(str(Path(__file__).resolve().parents[1].joinpath("src")))
import gewex_param as gwp
import gewex_variable as gwv
import gewex_netcdf as gwn


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
    choices=["Psurf", "temp", "h2o"],
    help=(
      "Variable to plot: \"Psurf\", \"temp\", \"h2o\""
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
    "-s", "--scatter", action="store_true",
    help="Use scatter plot instead of contour"
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
def read_f77(variable, filein, grid):

  with FortranFile(filein, "r", header_dtype=">u4") as f:
    rec = f.read_record(dtype=">f4").astype(dtype=np.float32)

  if variable.mode == "2d":
    shape = (grid.nlon, grid.nlat)
  else:
    if variable.name == "temp":
      shape = (grid.nlev+2, grid.nlon, grid.nlat)
    else:
      shape = (grid.nlev, grid.nlon, grid.nlat)

  # var_out = np.rollaxis(rec, -1, -2)
  # var_out = rec.reshape(grid.nlon, grid.nlat).T
  # var_out = np.rollaxis(rec.reshape(shape), -1, -2)
  var_out = rec.reshape(shape)
  # if variable.mode == "3d":
  #   var_out = np.rollaxis(var_out, 0, 3)
  var_out = np.rollaxis(var_out, -1, -2)

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
def init_figure(mode):

  A4R = (cm2inch(21.0), cm2inch(29.7))
  A4L = A4R[::-1]

  # print(A4R, A4L)

  if mode == "2d":
    figsize = A4R
    nrows = 3
    ncols = 1
    sharex = "none"
    sharey = "none"
  else:
    # figsize = A4R
    figsize = A4L
    nrows = 3
    # ncols = 1
    ncols = 3
    sharex = "all"
    sharey = "all"

  fig, axes = plt.subplots(
    figsize=figsize,
    nrows=nrows, ncols=ncols,
    sharex=sharex,
    sharey=sharey,
    # dpi=300,
  )

  return fig, axes


#----------------------------------------------------------------------
def prep_data(var_in, grid, fg_scatter=False):

  step = 4

  if fg_scatter:
    lon = grid.lon[::step]
    lat = grid.lat[::step]
    X = np.tile(lon, len(lat))
    Y = np.repeat(lat, len(lon))
    # X = np.tile(grid.lon[::4], int(grid.lat.size/4))
    # Y = np.repeat(grid.lat[::4], int(grid.lon.size/4))
    Z = var_in[::step, ::step].flatten()
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
  else:
    X = grid.lon  #[::4]
    Y = grid.lat  #[::4]
    Z = var_in

  # print(
  #   X.shape,
  #   Y.shape,
  #   Z.shape,
  # )

  return (X.copy(), Y.copy(), Z.copy())


#----------------------------------------------------------------------
def plot_data(fg_scatter, ax, X, Y, Z, cmap, norm, title, nb=30):

  if Z is not None:
    if fg_scatter:
      im = scatter_plot(ax, X, Y, Z, cmap, norm, title)
    else:
      im = contour_plot(ax, X, Y, Z, cmap, title, nb=nb)
    # im = ax.scatter(x=X, y=Y, c=Z, cmap=cmap, norm=norm)
    # cb = fig.colorbar(im, ax=ax, format="%.2e")
    write_comment(ax, Z)
    cb = fig.colorbar(im, ax=ax)
  else:
    cb = None
  ax.set_title(title, loc="left", fontsize=9)
  config_axcb(ax, cb, X, Y)


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
    0.98, 0.03,
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

  ax.tick_params(axis='both', which='major', labelsize=8)
  ax.tick_params(axis='x', labelrotation=45)


  if X is not None:
    cb.ax.tick_params(labelsize=8)
    cb.formatter.set_powerlimits((-2, 2))
    # cb.formatter.set_scientific(True)
    cb.update_ticks()

    # print(X.min(), X.max())
    # print(Y.min(), Y.max())
    # grid = tg_grid
    (ymin, xmin) = (floor(l) for l in (Y.min(), X.min()))
    (ymax, xmax) = (ceil(l)  for l in (Y.max(), X.max()))

    # cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
    # cbar = pyplot.colorbar(pp, orientation='vertical', ticks=np.arange(cbar_min, cbar_max+cbar_step, cbar_step), format=cbar_num_format)

    ax.set_xticks(range(xmin, xmax + 1, 30))
    ax.set_yticks(range(ymin, ymax + 1, 30))

    ax.set_xlim([float(xmin), float(xmin) + 360.])
    ax.set_ylim([float(ymin), float(ymax)])


#----------------------------------------------------------------------
def config_plot(fig, fig_title, now):

  plt.suptitle(fig_title)
  fig.text(
    0.98, 0.02,
    # "test",
    F"{now:%d/%m/%Y %H:%M:%S}",
    # F"{dt.datetime.now():%d/%m/%Y %H:%M:%S}",
    fontsize=8,
    fontstyle="italic",
    ha="right",
  )

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

  instru = gwp.InstruParam(args.runtype)
  params = gwp.GewexParam(project_dir)

  print(instru)
  print(params)


  variable = gwv.Variable(args.varname, instru)
  print(variable)

  if variable.mode == "2d":
    pl = {
       650: {"nc": 24, "tg": 16},
    }
  else:
    pl = {
        70: {"nc": 9, "tg": 0},
       100: {"nc": 10, "tg": 2},
       125: {"nc": 11, "tg": 3},
       175: {"nc": 13, "tg": 4},
       200: {"nc": 14, "tg": 5},
       225: {"nc": 15, "tg": 6},
       250: {"nc": 16, "tg": 7},
       300: {"nc": 17, "tg": 9},
       350: {"nc": 18, "tg": 10},
       650: {"nc": 24, "tg": 16},
       800: {"nc": 28, "tg": 18},
       850: {"nc": 30, "tg": 19},
       900: {"nc": 32, "tg": 20},
       950: {"nc": 34, "tg": 21},
      1000: {"nc": 36, "tg": 22},
      # 2000: {"nc": 36, "tg": 23},
      # 3000: {"nc": 36, "tg": 24},
    }

    if variable.name == "temp":
      pl[2000] = {"nc": 36, "tg": 23}
      pl[3000] = {"nc": 36, "tg": 24}

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
  nc_grid = gwn.NCGrid()
  nc_grid.load(variable.ncfiles)
  tg_grid = gwv.TGGrid()
  tg_grid.load(nc_grid)

  # pp.pprint(nc_grid.lev)

  # A4R = (cm2inch(21.0), cm2inch(29.7))
  # A4L = A4R[::-1]

  # print(A4R, A4L)

  # if variable.mode == "2d":
  #   figsize = A4R
  #   nrows = 3
  #   ncols = 1
  # else:
  #   # figsize = A4R
  #   figsize = A4L
  #   nrows = 3
  #   # ncols = 1
  #   ncols = 3

  # fig, axes = plt.subplots(
  #   figsize=figsize,
  #   nrows=nrows, ncols=ncols,
  #   # sharex="all",
  #   # sharey="all",
  #   # dpi=300,
  # )

  # cbars = {}
  # imags = {}

  # print(F"axes: {axes}")

  pdf = PdfPages(dir_img.joinpath(F"{img_name}.pdf"))


  print("Read input files")
  # ====================================================================

  print(" - netcdf")
  # ------------------------
  nstep = 24  # number of time steps per day
  i_time = (args.date.day - 1) * nstep
  variable.ncvalues = (  # NetCDF grid
      gwn.read_netcdf(variable, nc_grid, slice(nc_grid.nlon), i_time)
  )
  print(variable.ncvalues.shape)
  variable.ncvalues = (  # F77 grid
      gwv.grid_nc2tg(variable.ncvalues, nc_grid, tg_grid)
  )
  print(variable.ncvalues.shape)

  print(" - python")
  # ------------------------
  variable.pyvalues = \
      read_f77(variable, variable.pyfile, tg_grid)
      # read_f77(variable.pyfile, tg_grid)
  print(variable.pyvalues.shape)

  if fg_idl:
    print(" - idl")
    # ------------------------
    variable.idlvalues = \
        read_f77(variable, variable.idlfile, tg_grid)
        # read_f77(variable.idlfile, tg_grid)
    print(variable.idlvalues.shape)
  else:
    variable.idlvalues = []


  print("Prepare data")
  # ====================================================================

  # data = []
  # titles = []

  for ilev, (key, lev) in enumerate(pl.items()):

    if ilev % 3 == 0:
      print(F"{72*'~'}\nInit figure")
      fig, axes = init_figure(variable.mode)
      img_nb = 1 + (ilev // 3)
      fg_saved = False
      vmin = -np.inf
      vmax = +np.inf

    if lev["tg"] < tg_grid.nlev:
      print(
        F"{ilev}: {key} - "
        # F"{nc_grid.lev[lev['nc']]} hPa ; "
        # F"{tg_grid.lev[lev['tg']]} hPa"
      )
    else:
      print(
        F"{ilev}: {key} - "
        F"{lev['nc']} ; "
        F"{lev['tg']}"
      )

    print("Computing")
    if variable.mode == "2d":
      ncvalues = variable.ncvalues
      pyvalues = variable.pyvalues
      if fg_idl:
        idlvalues = variable.idlvalues

      level_nc = level_tg = "Surf"
    else:
      ncvalues = variable.ncvalues[lev["nc"], ...]
      pyvalues = variable.pyvalues[lev["tg"], ...]
      if fg_idl:
        idlvalues = variable.idlvalues[lev["tg"], ...]

      level_nc = F"{nc_grid.lev[lev['nc']]} hPa"
      if lev["tg"] < tg_grid.nlev:
        level_tg = F"{tg_grid.lev[lev['tg']]} hPa"
      else:
        level_tg = F"n+{1 + (tg_grid.nlev - lev['tg'])} hPa"

    if args.mode == "plot":
      norm = None
      titles = [
        F"netcdf @ {level_nc}",
        F"python @ {level_tg}",
      ]
      data = [ncvalues, pyvalues, ]
      if fg_idl:
        titles.append(F"idl @ {level_tg}", )
        data.append(idlvalues, )
      else:
        titles.append(F"no idl", )
        data.append(None, )
    else:
      norm = True
      titles = [
        F"(nc - py) @ ({level_nc} - {level_tg})",
      ]
      data = [ncvalues - pyvalues, ]
      if fg_idl:
        titles.extend((
          F"(nc - idl) @ ({level_nc} - {level_tg})",
          F"(idl - py) @ {level_tg}",
        ))
        data.extend((
          ncvalues - idlvalues,
          idlvalues - pyvalues,
        ))

    # if variable.mode == "3d":
    #   titles = [
    #       F"{t}     @ {nc_grid.lev[lev['nc']]} hPa"
    #       for t in titles
    #     ]


    for idx, (title, values) in enumerate(zip(titles, data)):
      iax = idx + 3 * (ilev % 3)

      print("Compute")
      if values is not None:
        print(title, values.shape)
        print(F" - [{idx + 3 * (ilev % 3)}] {title}")
        (X, Y, Z) = prep_data(values, tg_grid, args.scatter)
        print(F"mean = {Z.mean():.2e} ; std = {Z.std():.2e}")
        if norm:
          norm = clr.TwoSlopeNorm(vcenter=0., vmin=Z.min(), vmax=Z.max())
      else:
        X = Y = Z = None
      print(F"norm = {norm}")

      print("Plot stuff")
      plot_data(
        args.scatter, axes.T.flatten()[iax],
        X, Y, Z,
        cmap, norm,
        title
      )


    if (ilev % 3 == 2) or (ilev == len(pl)-1):
      print("Plot config")
      # ====================================================================
      now = dt.datetime.now()
      config_plot(fig, fig_title, now)

      print(
        F"Save fig {now:%d/%m/%Y %H:%M:%S} / "
        F"{img_name}_lev{img_nb}\n{72*'~'}"
      )
      # ========================
      if variable.mode == "2d":
        fileout = dir_img.joinpath(F"{img_name}.{img_type}"),
      else:
        fileout = dir_img.joinpath(F"{img_name}_lev{img_nb}.{img_type}"),

      fig.savefig(
        dir_img.joinpath(F"{img_name}_lev{img_nb}.{img_type}"),
      )
      fg_saved = True



  exit()

  for idx, (title, values) in enumerate(zip(titles, data)):
    # print(title, values.shape)
    print(F" - {title}")
    (X, Y, Z) = prep_data(values, tg_grid, args.scatter)
    print(F"mean = {Z.mean():.2e} ; std = {Z.std():.2e}")
    if norm:
      norm = clr.TwoSlopeNorm(vcenter=0., vmin=Z.min(), vmax=Z.max())
    print(F"norm = {norm}")

    plot_data(args.scatter, axes.T.flatten()[idx], X, Y, Z, cmap, norm, title)
    # plot_data(args.scatter, axes[idx], X, Y, Z, cmap, norm, title)

  # plt.show()

  # if args.mode == "plot":
  #   if variable.mode == "2d":
  #     ncvalues = variable.ncvalues
  #     pyvalues = variable.pyvalues
  #     idlvalues = variable.idlvalues
  #   else:
  #     ncvalues = variable.ncvalues[..., pl["650"]["nc"]]
  #     pyvalues = variable.pyvalues[..., pl["650"]["tg"]]
  #     idlvalues = variable.idlvalues[..., pl["650"]["tg"]]

  #   offset = 0.
  #   # if variable.name == "temp":
  #   #   offset = -273.15

  #   print(" - netcdf")
  #   # ------------------------
  #   (X0, Y0, Z0) = prep_data(ncvalues + offset, tg_grid, args.scatter)
  #   print(Z0.mean(), Z0.std())
  #   norm0 = None
  #   title0 = "netcdf"
  #   print(" - python")
  #   # ------------------------
  #   (X1, Y1, Z1) = prep_data(pyvalues + offset, tg_grid, args.scatter)
  #   print(Z1.mean(), Z1.std())
  #   norm1 = None
  #   title1 = "python"
  #   if fg_idl:
  #     print(" - idl")
  #     # ------------------------
  #     (X2, Y2, Z2) = prep_data(idlvalues + offset, tg_grid, args.scatter)
  #     print(Z2.mean(), Z2.std())
  #     norm2 = None
  #     title2 = "idl"
  # else:
  #   print(" - (netcdf - python)")
  #   # ------------------------
  #   (X0, Y0, Z0) = prep_data(
  #     variable.ncvalues - variable.pyvalues, tg_grid, args.scatter
  #   )
  #   print(Z0.mean(), Z0.std())
  #   norm0 = clr.TwoSlopeNorm(vcenter=0., vmin=Z0.min(), vmax=Z0.max())
  #   title0 = "nc - py"
  #   if fg_idl:
  #     print(" - (netcdf - idl)")
  #     # ------------------------
  #     (X1, Y1, Z1) = prep_data(
  #       variable.ncvalues - variable.idlvalues, tg_grid, args.scatter
  #     )
  #     print(Z1.mean(), Z1.std())
  #     norm1 = clr.TwoSlopeNorm(vcenter=0., vmin=Z1.min(), vmax=Z1.max())
  #     title1 = "nc - idl"
  #     print(" - (idl - python)")
  #     # ------------------------
  #     (X2, Y2, Z2) = prep_data(
  #       variable.idlvalues - variable.pyvalues, tg_grid, args.scatter
  #     )
  #     print(Z2.mean(), Z2.std())
  #     norm2 = clr.TwoSlopeNorm(vcenter=0., vmin=Z2.min(), vmax=Z2.max())
  #     title2 = "idl - py"


  # print("Plot data")
  # # ====================================================================
  # plot_data(args.scatter, axes[0], X0, Y0, Z0, cmap, norm0, title0)
  # plot_data(args.scatter, axes[1], X1, Y1, Z1, cmap, norm1, title1)
  # plot_data(args.scatter, axes[2], X2, Y2, Z2, cmap, norm2, title2)


  print("Plot config")
  # ====================================================================
  now = dt.datetime.now()
  config_plot(fig, fig_title, now)

  print(F"Save fig {now:%d/%m/%Y %H:%M:%S} / {img_name}")
  # ========================

  pdf.savefig(dpi=75)

  pdf.close()

  # plt.show()
  fig.savefig(
    # "P_plot.png",
    # dir_img.joinpath(img_name),
    dir_img.joinpath(F"{img_name}.{img_type}"),
    # bbox_inches="tight",
  )

