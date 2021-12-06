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
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import colors as clr
# from matplotlib.backends.backend_pdf import PdfPages
# from mpl_toolkits.basemap import Basemap
# from netCDF4 import Dataset

import pprint
pp = pprint.PrettyPrinter(indent=2)

# Application library imports
# ========================
sys.path.append(str(Path(__file__).resolve().parents[1].joinpath("src")))
import gwx_param as gwp
import gwx_variable as gwv
import gwx_netcdf as gwn


#######################################################################
def get_arguments():
  from argparse import ArgumentParser
  from argparse import RawTextHelpFormatter

  parser = ArgumentParser(
    formatter_class=RawTextHelpFormatter
  )

  parser.add_argument(
    "varname", action="store",
    # type=int,
    choices=["P", "T", "Q", "S"],
    help=(
      "Variable to check"
    )
  )
  parser.add_argument(
    "runtype", action="store",
    type=int,
    choices=[1, 2, 3, 4, 5, 6, 9],
    help=(
      "Run type:\n"
      "  - 1 = AIRS / AM\n"
      "  - 2 = AIRS / PM\n"
      "  - 3 = IASI / AM\n"
      "  - 4 = IASI / PM\n"
      "  - 5 = IASI-B / AM\n"
      "  - 6 = IASI-B / PM\n"
      "  - 9 = Test mode (node = 0.0)\n"
    )
  )
  parser.add_argument(
    "date_start", action="store",
    type=lambda s: dt.datetime.strptime(s, "%Y%m%d"),
    help="Start date: YYYYMMJJ"
  )
  parser.add_argument(
    "date_end", action="store",
    type=lambda s: dt.datetime.strptime(s, "%Y%m%d"),
    help="End date: YYYYMMJJ"
  )

  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )

  return parser.parse_args()


#----------------------------------------------------------------------
# def cm2inch(x):

#   return x / 2.54


#----------------------------------------------------------------------
def read_f77(variable, filein, grid):

  # print(variable.name)

  if variable.name == "surftype":
    dtype_in = ">i4"
    dtype_out = np.int32
  else:
    dtype_in = ">f4"
    dtype_out = np.float32

  with FortranFile(filein, "r", header_dtype=">u4") as f:
    rec = f.read_record(dtype=dtype_in).astype(dtype=dtype_out)
    # rec = f.read_record(dtype=">f4").astype(dtype=np.float32)

  if variable.mode == "2d":
    shape = (grid.nlon, grid.nlat)
  else:
    shape = (grid.nlev + V.extralev, grid.nlon, grid.nlat)
    # if variable.name == "temp":
    #   shape = (grid.nlev+2, grid.nlon, grid.nlat)
    # else:
    #   shape = (grid.nlev, grid.nlon, grid.nlat)

  var_out = rec.reshape(shape)
  var_out = np.rollaxis(var_out, -1, -2)

  return var_out


# #----------------------------------------------------------------------
# def pl_dict(variable):

#   if variable.mode == "2d":
#     pl = {
#        "surf": {"nc": None, "tg": None},
#     }
#   else:
#     pl = {
#         70: {"nc": 9, "tg": 0},
#        100: {"nc": 10, "tg": 2},
#        125: {"nc": 11, "tg": 3},
#        175: {"nc": 13, "tg": 4},
#        200: {"nc": 14, "tg": 5},
#        225: {"nc": 15, "tg": 6},
#        250: {"nc": 16, "tg": 7},
#        300: {"nc": 17, "tg": 9},
#        350: {"nc": 18, "tg": 10},
#        650: {"nc": 24, "tg": 16},
#        800: {"nc": 28, "tg": 18},
#        850: {"nc": 30, "tg": 19},
#        900: {"nc": 32, "tg": 20},
#        950: {"nc": 34, "tg": 21},
#       1000: {"nc": 36, "tg": 22},
#       # 2000: {"nc": 36, "tg": 23},
#       # 3000: {"nc": 36, "tg": 24},
#     }

#     if variable.name == "temp":
#       pl[2000] = {"nc": 36, "tg": 23}
#       pl[3000] = {"nc": 36, "tg": 24}

#   return pl


# #----------------------------------------------------------------------
# def img_name(mode, variable, date, instru):

#   return (
#     F"{args.mode}_"
#     F"{variable.name}_{date:%Y%m%d}_"
#     F"{instru.name}_{instru.ampm}"
#   )


# #----------------------------------------------------------------------
# def fig_title(variable, date, instru):

#   return (
#     F"{variable.longname} ({variable.units})\n"
#     F"{date:%d/%m/%Y} "
#     F"{instru.name}_{instru.ampm}"
#   )


# #----------------------------------------------------------------------
# def get_titles(mode, variable, key, lev, ncgrid, tggrid):

#   if key == "surf":
#     level_nc = level_tg = key
#   else:
#     level_nc = F"{ncgrid.lev[lev['nc']]} hPa"
#     if lev["tg"] < tggrid.nlev:
#       level_tg = F"{tggrid.lev[lev['tg']]} hPa"
#     else:
#       level_tg = F"surf (n+{1 + (lev['tg'] - tggrid.nlev)})"

#   if mode == "plot":
#     titles = [
#       F"netcdf @ {level_nc}",
#       F"python @ {level_tg}",
#       F"idl @ {level_tg}",
#     ]
#   else:
#     titles = [
#       F"(nc - py) @ ({level_nc} - {level_tg})",
#       F"(nc - idl) @ ({level_nc} - {level_tg})",
#       F"(idl - py) @ {level_tg}",
#     ]

#   return titles


# #----------------------------------------------------------------------
# def get_datasets(mode, variable, key, lev, ncgrid, tggrid):

#   if variable.mode == "2d":
#     ncvalues = variable.ncvalues
#     pyvalues = variable.pyvalues
#     idlvalues = variable.idlvalues

#     # level_nc = level_tg = key
#   else:
#     ncvalues = variable.ncvalues[lev["nc"], ...]
#     pyvalues = variable.pyvalues[lev["tg"], ...]
#     if variable.idlvalues.any():
#       idlvalues = variable.idlvalues[lev["tg"], ...]
#     else:
#       idlvalues = np.empty((0, 0), dtype=np.float)

#   if mode == "plot":
#     data0 = ncvalues
#     data1 = pyvalues
#     data2 = idlvalues
#   else:
#     data0 = ncvalues - pyvalues
#     if variable.idlvalues.any():
#       data1 = ncvalues - idlvalues
#       data2 = idlvalues - pyvalues
#     else:
#       data1 = data2 = np.empty((0, 0), dtype=np.float)

#   return [data0, data1, data2]


# #----------------------------------------------------------------------
# def get_norms(mode, variable, datasets):

#   norms = []

#   # print([data.max() for data in datasets if data.any()])
#   # print(np.max([data.max() for data in datasets if data.any()]))
#   # print([data.min() for data in datasets if data.any()])
#   # print(np.min([data.min() for data in datasets if data.any()]))

#   vmin = np.min([data.min() for data in datasets if data.any()])
#   vmax = np.max([data.max() for data in datasets if data.any()])

#   print(vmin, vmax)

#   for data in datasets:
#     print(data.shape)
#     if data.any():
#       # print(data.min(), data.max())
#       if mode == "plot":
#         # norm = clr.Normalize(data.min(), data.max())
#         norm = clr.Normalize(vmin, vmax)
#         norms.append(norm)
#       else:
#         vmax = np.max((abs(data.min()), abs(data.max())))
#         # norm = clr.CenteredNorm()
#         norm = clr.TwoSlopeNorm(vcenter=0., vmin=-vmax, vmax=vmax)
#         norms.append(norm)
#     else:
#       norms.append(None)

#   return norms


# #----------------------------------------------------------------------
# def init_figure(mode, nocoast):

#   # import cartopy.crs as ccrs

#   A4R = (cm2inch(21.0), cm2inch(29.7))
#   A4L = A4R[::-1]

#   if mode == "2d":
#     figsize = A4R
#     ngs = 1
#     wspace, hspace = (0.05, 0.25)
#     lrs =(
#         (0.075, 0.925, ),
#       )
#   else:
#     figsize = A4L
#     ngs = 3
#     wspace, hspace = (0.05, 0.40)
#     lrs =(
#       (0.050, 0.300, ),
#       (0.375, 0.625, ),
#       (0.700, 0.950, ),
#     )

#   fig = plt.figure(
#     figsize=figsize,
#     # sharex = "all",
#     # sharey = "all",
#     # dpi=300,  # 100
#   )

#   nrows, ncols = (3, 2)
#   bottom, top = (0.08, 0.90)
#   # wspace, hspace = (0.05, 0.40)
#   heights = [1, 1, 1, ]
#   widths = [20, 1, ]

#   gs_kw = dict(
#     width_ratios=widths,
#     height_ratios=heights,
#     bottom=bottom,
#     top=top,
#     wspace=wspace,
#     hspace=hspace,
#   )

#   axes = []
#   caxes = []
#   # crs = ccrs.PlateCarree(central_longitude=0)

#   for left, right in lrs:
#     print(right, left)
#     gs_kw["left"] = left
#     gs_kw["right"] = right

#     gs = fig.add_gridspec(
#       nrows=nrows,
#       ncols=ncols,
#       **gs_kw,
#     )

#     for r in range(nrows):
#       ax = fig.add_subplot(
#         gs[r, 0],
#         # projection=crs
#       )
#       # if not nocoast:
#         # ax.coastlines(linewidth=0.25)

#       cax = fig.add_subplot(gs[r, 1])
#       axes.append(ax)
#       caxes.append(cax)

#   return fig, axes, caxes


# #----------------------------------------------------------------------
# def set_titles(ax, title):

#   ax.set_title(title, loc="left", fontsize=10)


# #----------------------------------------------------------------------
# def config_axes(axes):

#   for ax in axes:
#     ax.tick_params(axis="both", which="major", labelsize=8)
#     ax.tick_params(axis="x", labelrotation=30)
#     ax.set_xticks(range(-180, 180 + 1, 30))
#     ax.set_yticks(range(-90, 90 + 1, 30))
#     ax.set_aspect("auto")


# #----------------------------------------------------------------------
# def config_caxes(caxes):

#   for ax in caxes:
#     ax.tick_params(
#         axis="both",        # changes apply to the x-axis
#         which="both",       # both major and minor ticks are affected
#         bottom=False,       # ticks along the bottom edge are off
#         left=False,         # ticks along the bottom edge are off
#         labelbottom=False,  # labels along the bottom edge are off
#         labelleft=False,    # labels along the bottom edge are off
#     )


# #----------------------------------------------------------------------
# def plot_data(ax, cax, data, norm, cmap):


#   if data.any():
#     print("data")
#     im = ax.imshow(
#       data,
#       extent=[-180., 180., -90., 90.],
#       norm=norm,
#       cmap=cmap,
#       origin="lower",
#       aspect="auto",
#     )
#     cb = fig.colorbar(im, cax=cax)
#     cb.ax.tick_params(labelsize=8)
#     cb.formatter.set_powerlimits((-2, 2))
#     cb.update_ticks()

#   else:
#     print("vide")
#     im = ax.plot(
#       [0.0, 1.0, 0.0, 1.0],
#       [0.0, 1.0, 1.0, 0.0],
#       "black",
#       linewidth=0.5,
#       transform=ax.transAxes,
#     )


# #----------------------------------------------------------------------
# def write_comment(ax, data):

#   if data.any():
#     ax.text(
#       0.98, 0.04,
#       F"m = {data.mean():.2e} / $\sigma$ = {data.std():.2e}",
#       fontsize=8,
#       ha="right",
#       va="center",
#       transform=ax.transAxes,
#     )
#     ax.text(
#       0.98, 0.96,
#       # F"min = {data.min():.2e} / max = {data.max():.2e}",
#       F"{data.min():.2e} < $\Delta$ < {data.max():.2e}",
#       fontsize=8,
#       ha="right",
#       va="center",
#       transform=ax.transAxes,
#     )


# #----------------------------------------------------------------------
# def config_plot(fig, fig_title, now):

#   plt.suptitle(fig_title)

#   fig.text(
#     0.98, 0.02,
#     # "test",
#     F"{now:%d/%m/%Y %H:%M:%S}",
#     # F"{dt.datetime.now():%d/%m/%Y %H:%M:%S}",
#     fontsize=8,
#     fontstyle="italic",
#     ha="right",
#   )


#######################################################################

if __name__ == "__main__":

  # .. Initialization ..
  # ====================
  # ... Command line arguments ...
  # ------------------------------
  args = get_arguments()
  if args.verbose:
    print(args)

  # ... Constants ...
  # -----------------
  fileversion = "SL04"

  # ... Files and directories ...
  # -----------------------------
  project_dir = Path(__file__).resolve().parents[1]

  # .. Main program ..
  # ==================

  # ... Initialize things ...
  # -------------------------
  instru = gwp.InstruParam(args.runtype)
  params = gwp.GewexParam(project_dir)

  if args.verbose:
    print(instru)
    print(params)

  # Grids
  tggrid = gwv.TGGrid()
  tggrid.load()

  # Variables
  V = gwv.VarOut(args.varname, instru, fileversion)

  print(V.longname, instru.name, instru.ampm)
  print(
    F"{20*' '}   min  "
    F"   max  "
    F"  mean  "
    F"   std  "
    F"     neg  "
    F"  zeroes  "
    F"     pos  "
  )

  pattern = F"**/unmasked_ERA5_{instru.name}_{V.outstr}.*.{instru.ampm}_{fileversion}"
  for f in sorted(params.dirout.glob(pattern)):
    V.pyvalues = read_f77(V, f, tggrid)

    # cond = (V.pyvalues == 0.)
    # nb_zeroes = np.count_nonzero(cond)
    # cond = (V.pyvalues < 0.)
    # nb_neg = np.count_nonzero(cond)
    # cond = (V.pyvalues > 0.)
    # nb_pos = np.count_nonzero(cond)

    (nb_zeroes, nb_neg, nb_pos) = (
      np.count_nonzero(cond)
        for cond in ((V.pyvalues == 0.), (V.pyvalues < 0.), (V.pyvalues > 0.))
    )

    print(
      F"{''.join(f.suffixes)} : "
      F"{V.pyvalues.min():6.2f}  "
      F"{V.pyvalues.max():6.2f}  "
      F"{V.pyvalues.mean():6.2f}  "
      F"{V.pyvalues.std():6.2f}  "
      F"{nb_neg:8d}  "
      F"{nb_zeroes:8d}  "
      F"{nb_pos:8d}  "
    )

    # print(f)
    # print(V.pyvalues.shape)
    # print(
    #   F"min: {V.pyvalues.min()} ; "
    #   F"max: {V.pyvalues.max()} ; "
    #   F"mean: {V.pyvalues.mean()} ; "
    #   F"std: {V.pyvalues.std()} ; "
    # )

  # print(nb_neg, nb_zeroes, nb_pos)

  exit()
