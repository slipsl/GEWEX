#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
# ========================
import sys
from pathlib import Path
# import os.path
from math import floor, ceil
import datetime as dt
import numpy as np
from scipy.io import FortranFile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as clr
# from matplotlib.backends.backend_pdf import PdfPages
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
    choices=["Psurf", "temp", "h2o", "surftype"],
    help=(
      "Variable to plot"
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
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )

  parser.add_argument(
    "--pyversion", action="store",
    default="SL04",
    help="File version (default = \"SL04\")"
  )

  parser.add_argument(
    "--idlversion", action="store",
    default="06",
    help="File version (default = \"06\")"
  )

  parser.add_argument(
    "--nocoast", action="store_true",
    help="Don't draw coastlines"
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

  var_out = rec.reshape(shape)
  var_out = np.rollaxis(var_out, -1, -2)

  return var_out


#----------------------------------------------------------------------
def pl_dict(variable):

  if variable.mode == "2d":
    pl = {
       "surf": {"nc": None, "tg": None},
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

  return pl


#----------------------------------------------------------------------
def img_name(mode, variable, date, instru):

  return (
    F"{args.mode}_"
    F"{variable.name}_{date:%Y%m%d}_"
    F"{instru.name}_{instru.ampm}"
  )


#----------------------------------------------------------------------
def fig_title(variable, date, instru):

  return (
    F"{variable.longname} ({variable.units})\n"
    F"{date:%d/%m/%Y} "
    F"{instru.name}_{instru.ampm}"
  )


#----------------------------------------------------------------------
def get_titles(mode, variable, key, lev, ncgrid, tggrid):

  if key == "surf":
    level_nc = level_tg = key
  else:
    level_nc = F"{ncgrid.lev[lev['nc']]} hPa"
    if lev["tg"] < tggrid.nlev:
      level_tg = F"{tggrid.lev[lev['tg']]} hPa"
    else:
      level_tg = F"surf (n+{1 + (lev['tg'] - tggrid.nlev)})"

  if mode == "plot":
    titles = [
      F"netcdf @ {level_nc}",
      F"python @ {level_tg}",
      F"idl @ {level_tg}",
    ]
  else:
    titles = [
      F"(nc - py) @ ({level_nc} - {level_tg})",
      F"(nc - idl) @ ({level_nc} - {level_tg})",
      F"(idl - py) @ {level_tg}",
    ]

  return titles


#----------------------------------------------------------------------
def get_datasets(mode, variable, key, lev, ncgrid, tggrid):

  if variable.mode == "2d":
    ncvalues = variable.ncvalues
    pyvalues = variable.pyvalues
    idlvalues = variable.idlvalues

    # level_nc = level_tg = key
  else:
    ncvalues = variable.ncvalues[lev["nc"], ...]
    pyvalues = variable.pyvalues[lev["tg"], ...]
    if variable.idlvalues.any():
      idlvalues = variable.idlvalues[lev["tg"], ...]
    else:
      idlvalues = np.empty((0, 0), dtype=np.float)

  if mode == "plot":
    data0 = ncvalues
    data1 = pyvalues
    data2 = idlvalues
  else:
    data0 = ncvalues - pyvalues
    if variable.idlvalues.any():
      data1 = ncvalues - idlvalues
      data2 = idlvalues - pyvalues
    else:
      data1 = data2 = np.empty((0, 0), dtype=np.float)

  return [data0, data1, data2]


#----------------------------------------------------------------------
def get_norms(mode, variable, datasets):

  norms = []

  # print([data.max() for data in datasets if data.any()])
  # print(np.max([data.max() for data in datasets if data.any()]))
  # print([data.min() for data in datasets if data.any()])
  # print(np.min([data.min() for data in datasets if data.any()]))

  vmin = np.min([data.min() for data in datasets if data.any()])
  vmax = np.max([data.max() for data in datasets if data.any()])

  print(vmin, vmax)

  for data in datasets:
    print(data.shape)
    if data.any():
      # print(data.min(), data.max())
      if mode == "plot":
        # norm = clr.Normalize(data.min(), data.max())
        norm = clr.Normalize(vmin, vmax)
        norms.append(norm)
      else:
        vmax = np.max((abs(data.min()), abs(data.max())))
        # norm = clr.CenteredNorm()
        norm = clr.TwoSlopeNorm(vcenter=0., vmin=-vmax, vmax=vmax)
        norms.append(norm)
    else:
      norms.append(None)

  return norms


#----------------------------------------------------------------------
def init_figure(mode, nocoast):

  import cartopy.crs as ccrs

  A4R = (cm2inch(21.0), cm2inch(29.7))
  A4L = A4R[::-1]

  if mode == "2d":
    figsize = A4R
    ngs = 1
    wspace, hspace = (0.05, 0.25)
    lrs =(
        (0.075, 0.925, ),
      )
  else:
    figsize = A4L
    ngs = 3
    wspace, hspace = (0.05, 0.40)
    lrs =(
      (0.050, 0.300, ),
      (0.375, 0.625, ),
      (0.700, 0.950, ),
    )

  fig = plt.figure(
    figsize=figsize,
    # sharex = "all",
    # sharey = "all",
    # dpi=300,  # 100
  )

  nrows, ncols = (3, 2)
  bottom, top = (0.08, 0.90)
  # wspace, hspace = (0.05, 0.40)
  heights = [1, 1, 1, ]
  widths = [20, 1, ]

  gs_kw = dict(
    width_ratios=widths,
    height_ratios=heights,
    bottom=bottom,
    top=top,
    wspace=wspace,
    hspace=hspace,
  )

  axes = []
  caxes = []
  crs = ccrs.PlateCarree(central_longitude=0)

  for left, right in lrs:
    print(right, left)
    gs_kw["left"] = left
    gs_kw["right"] = right

    gs = fig.add_gridspec(
      nrows=nrows,
      ncols=ncols,
      **gs_kw,
    )

    for r in range(nrows):
      ax = fig.add_subplot(
        gs[r, 0],
        projection=crs
      )
      if not nocoast:
        ax.coastlines(linewidth=0.25)

      cax = fig.add_subplot(gs[r, 1])
      axes.append(ax)
      caxes.append(cax)

  return fig, axes, caxes


#----------------------------------------------------------------------
def set_titles(ax, title):

  ax.set_title(title, loc="left", fontsize=10)


#----------------------------------------------------------------------
def config_axes(axes):

  for ax in axes:
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_xticks(range(-180, 180 + 1, 30))
    ax.set_yticks(range(-90, 90 + 1, 30))
    ax.set_aspect("auto")


#----------------------------------------------------------------------
def config_caxes(caxes):

  for ax in caxes:
    ax.tick_params(
        axis="both",        # changes apply to the x-axis
        which="both",       # both major and minor ticks are affected
        bottom=False,       # ticks along the bottom edge are off
        left=False,         # ticks along the bottom edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False,    # labels along the bottom edge are off
    )


#----------------------------------------------------------------------
def plot_data(ax, cax, data, norm, cmap):


  if data.any():
    print("data")
    im = ax.imshow(
      data,
      extent=[-180., 180., -90., 90.],
      norm=norm,
      cmap=cmap,
      origin="lower",
      aspect="auto",
    )
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()

  else:
    print("vide")
    im = ax.plot(
      [0.0, 1.0, 0.0, 1.0],
      [0.0, 1.0, 1.0, 0.0],
      "black",
      linewidth=0.5,
      transform=ax.transAxes,
    )


#----------------------------------------------------------------------
def write_comment(ax, data):

  if data.any():
    ax.text(
      0.98, 0.04,
      F"m = {data.mean():.2e} / $\sigma$ = {data.std():.2e}",
      fontsize=8,
      ha="right",
      va="center",
      transform=ax.transAxes,
    )
    ax.text(
      0.98, 0.96,
      # F"min = {data.min():.2e} / max = {data.max():.2e}",
      F"{data.min():.2e} < $\Delta$ < {data.max():.2e}",
      fontsize=8,
      ha="right",
      va="center",
      transform=ax.transAxes,
    )


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
  dir_idl = project_dir.joinpath("output", "exemples")
  dir_img = project_dir.joinpath("img")

  cmap_plot = "coolwarm"
  cmap_diff = "seismic"

  if args.mode == "plot":
    cmap = cmap_plot
  else:
    cmap = cmap_diff

  instru = gwp.InstruParam(args.runtype)
  params = gwp.GewexParam(project_dir)

  print(instru)
  print(params)

  variable = gwv.Variable(args.varname, instru, args.pyversion)
  print(variable)

  # print(variable.get_ncfiles(params.dirin, args.date))
  # print(variable.pathout(params.dirout, args.date))
  # print(variable.dirout(params.dirout, args.date))
  # print(variable.fileout(args.date))

  img_type = "png"
  img_name = img_name(args.mode, variable, args.date, instru)
  print(F"img = {img_name}")

  fig_title = fig_title(variable, args.date, instru)

  variable.ncfiles = variable.get_ncfiles(params.dirin, args.date)
  variable.pyfile = variable.pathout(params.dirout, args.date)
  variable.idlfile = dir_idl.joinpath(variable.fileout(args.date))

  variable.idlfile = Path(
    # str(variable.idlfile).replace(F"_{args.pyversion}", "_05")
    str(variable.idlfile).replace(F"_{args.pyversion}", F"_{args.idlversion}")
  )

  pp.pprint(
    tuple(str(f) for f in (variable.ncfiles, variable.pyfile, variable.idlfile))
  )

  if not variable.idlfile.exists():
    fg_idl = False
  else:
    fg_idl = True

  # ... Load NetCDF & target grids ...
  # ----------------------------------
  ncgrid = gwn.NCGrid()
  ncgrid.load(variable.ncfiles)
  tggrid = gwv.TGGrid()
  tggrid.load()

  # pdf = PdfPages(dir_img.joinpath(F"{img_name}.pdf"))

  print("Read input files")
  # ====================================================================

  print(" - netcdf")
  # ------------------------
  nstep = 24  # number of time steps per day
  t = (args.date.day - 1) * nstep + int(instru.tnode)
  variable.ncvalues = (  # NetCDF grid
    gwn.read_netcdf_t(variable, t)
  )
  # print(variable.ncvalues.shape)
  variable.ncvalues = (  # F77 grid
    gwv.grid_nc2tg(variable.ncvalues, ncgrid, tggrid)
  )
  print(variable.ncvalues.shape)

  print(" - python")
  # ------------------------
  variable.pyvalues = \
      read_f77(variable, variable.pyfile, tggrid)
  print(variable.pyvalues.shape)

  if fg_idl:
    print(" - idl")
    # ------------------------
    variable.idlvalues = \
        read_f77(variable, variable.idlfile, tggrid)
    # print(variable.idlvalues.shape)
  else:
    # variable.idlvalues = []
    variable.idlvalues = np.empty(shape=(0, 0), dtype=np.float)

  print(variable.idlvalues.shape)
  print(variable.pyvalues.any())
  print(variable.idlvalues.any())


  print("Prepare data")
  # ====================================================================

  # data = []
  # titles = []

  pl = pl_dict(variable)

  for ilev, (key, lev) in enumerate(pl.items()):
    if ilev % 3 == 0:
      print(F"{72*'~'}\nInit figure")
      fig, axes, caxes = init_figure(variable.mode, args.nocoast)
      img_nb = 1 + (ilev // 3)

    print(ilev, key, lev)

    print("Define titles")
    titles = get_titles(args.mode, variable, key, lev, ncgrid, tggrid)
    # print("titles: ", titles)

    print("Define datasets")
    datasets = get_datasets(args.mode, variable, key, lev, ncgrid, tggrid)
    # print("datasets: ", datasets)

    print("Define norms")
    norms = get_norms(args.mode, variable, datasets)
    # print("norms: ", norms)

    for idx, (title, data, norm) in enumerate(zip(titles, datasets, norms)):
      iax = idx + 3 * (ilev % 3)

      print("Plot data / empty")
      plot_data(axes[iax], caxes[iax], data, norm, cmap)
      set_titles(axes[iax], title)
      write_comment(axes[iax], data)

    if (ilev % 3 == 2) or (ilev == len(pl)-1):
      print("Plot config")
      config_axes(axes)
      config_caxes(caxes)
      now = dt.datetime.now()
      config_plot(fig, fig_title, now)

      print(
        F"Save fig {now:%d/%m/%Y %H:%M:%S} / "
        F"{img_name}_lev{img_nb}\n{72*'~'}"
      )
      if variable.mode == "2d":
        fileout = dir_img.joinpath(F"{img_name}.{img_type}")
      else:
        fileout = dir_img.joinpath(F"{img_name}_lev{img_nb}.{img_type}")

      fig.savefig(fileout)

  # plt.show()
