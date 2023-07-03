#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==================================================================== #
# Author: Sonia Labetoulle                                             #
# Contact: sonia.labetoulle _at_ ipsl.jussieu.fr                       #
# Created: 2019                                                        #
# History:                                                             #
# Modification:                                                        #
# ==================================================================== #

# Standard library imports
# ========================
import os
from pathlib import Path
import socket
import subprocess
import datetime as dt
import pprint
from argparse import Action
import collections

pp = pprint.PrettyPrinter(indent=2)

import gwx_param as gwp



#######################################################################
# class ValidateParam(argparse.Action):
class ValidateParam(Action):
  def __call__(self, parser, args, values, option_string=None):
    # print(F"{args} {values} {option_string}")
    valid_runtypes = ("0", "1", "2", "3", "4", "5", "6", "9")
    runtype, year = values
    if runtype not in valid_runtypes:
      raise ValueError('invalid subject {s!r}'.format(s=runtype))
    year = dt.datetime.strptime(year, "%Y")
    Param = collections.namedtuple("Param", "runtype year")
    setattr(args, self.dest, Param(runtype, year))


#######################################################################
class bcolors:
  NoColor    = "\033[0m"
  Black      = "\033[0;30m"
  ULine      = "\033[4m"
  Bold       = "\033[1m"
  Red        = "\033[0;31m"
  BoldRed    = "\033[1;31m"
  Green      = "\033[0;32m"
  BoldGreen  = "\033[1;32m"
  Brown      = "\033[0;33m"
  BoldBrown  = "\033[1;33m"                                                                                   
  Blue       = "\033[0;34m"
  BoldBlue   = "\033[1;34m" 
  Purple     = "\033[0;35m"
  BoldPurple = "\033[1;35m" 
  Cyan       = "\033[0;36m"
  BoldCyan   = "\033[1;36m" 
  White      = "\033[0;37m"
  BoldWhite  = "\033[1;37m" 


#######################################################################
def get_arguments():
  from argparse import ArgumentParser
  from argparse import RawTextHelpFormatter
  # from argparse import ArgumentDefaultsHelpFormatter

  parser = ArgumentParser(
    formatter_class=RawTextHelpFormatter
    # formatter_class=ArgumentDefaultsHelpFormatter
  )

  # parser.add_argument(
  #   "runtype", action="store",
  #   type=int,
  #   choices=[1, 2, 3, 4, 5, 6, 9],
  #   help=(
  #     "Run type:\n"
  #     "  - 1 = AIRS / AM\n"
  #     "  - 2 = AIRS / PM\n"
  #     "  - 3 = IASI-A / AM\n"
  #     "  - 4 = IASI-A / PM\n"
  #     "  - 5 = IASI-B / AM\n"
  #     "  - 6 = IASI-B / PM\n"
  #     "  - 9 = Test mode (node = 0.0)\n"
  #   )
  # )
  # parser.add_argument(
  #   "year", action="store",
  #   type=lambda s: dt.datetime.strptime(s, "%Y"),
  #   help="Year: YYYY"
  # )

  parser.add_argument(
    "-p", "--param", nargs=2, 
    # action="store",
    action=ValidateParam,
    help=(
      "(Run type, Year YYYY)\n\n"
      "Run type:\n"
      "  - 0 = All\n"
      "  - 1 = AIRS / AM\n"
      "  - 2 = AIRS / PM\n"
      "  - 3 = IASI-A / AM\n"
      "  - 4 = IASI-A / PM\n"
      "  - 5 = IASI-B / AM\n"
      "  - 6 = IASI-B / PM\n"
      "  - 9 = Test mode (node = 0.0)\n"
    ),
    metavar=('Runtype', 'Year')
  )

  parser.add_argument(
    "--version", action="store",
    default="SL09",
    help="File version, default value \"%(default)s\""
  )
  # parser.add_argument(
  #   "-r", "--remove", action="store_true",
  #   help="Remove duplicate file"
  # )
  # parser.add_argument(
  #   "--notemp", action="store_true",
  #   help="Don't produce temperature files"
  # )
  # parser.add_argument(
  #   "--noh2o", action="store_true",
  #   help="Don't produce spec. humidity files"
  # )
  # parser.add_argument(
  #   "--nosurf", action="store_true",
  #   help="Don't produce surface type files"
  # )

  parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Verbose mode"
  )

  return parser.parse_args()


#######################################################################
def get_params(jobname):

  (t, d1, d2) = jobname.split("_")

  runtype = int(t[-1])
  instru = gwp.InstruParam(runtype)

  if len(d1) == 2:
    year = dt.datetime.strptime(d1, "%y")
    bornes = (
      dt.datetime.strptime(F"{d1}{d2[:4]}", "%y%m%d"),
      dt.datetime.strptime(F"{d1}{d2[4:]}", "%y%m%d")
    )
  else:
    year = dt.datetime.strptime(d1[0:2], "%y")
    bornes = (
      dt.datetime.strptime(d1, "%y%m%d"),
      dt.datetime.strptime(d2, "%y%m%d")
    )

  # pattern = F"unmasked_ERA5_{instru.name}_*.{year:%Y}*.{instru.ampm}_{args.version}"

  return instru, year, bornes  # , pattern


#######################################################################
def get_pattern(instru, year):

  pattern = F"unmasked_ERA5_{instru.name}_*.{year:%Y}*.{instru.ampm}_{args.version}"

  return pattern


#######################################################################
def get_files(year, bornes, pattern):

  nb_file = 0
  last_file = None

  p = dirin.joinpath(F"{year:%Y}")
  for dirname in sorted(x for x in p.iterdir() if x.is_dir()):
    if (bornes[0].month <= int(dirname.name) <= bornes[1].month):
      nb_file = nb_file + len(list(dirname.glob(pattern)))
      if len(list(dirname.glob(pattern))) > 0:
        last_file = sorted(list(dirname.glob(pattern)))[-1]

  return last_file, nb_file


#######################################################################
def parse_proc(proc, fg_spirit):

  if fg_spirit:
    jobs = parse_squeue(proc)
  else:
    jobs = parse_qsub(proc)

  return jobs


#######################################################################
def parse_qsub(proc):

  jobs = []
  job = dict()

  nb_data = 0
  jobid = None
  jobname = None
  jobstate = None
  timeused = None
  timereqd = None
  
  for l in proc.stdout.split(b"\n"):
    l = str(l, "UTF8")

    # if b"Job Id" in l:
    if "Job Id" in l:
      if args.verbose:
        print(l)
      jobid = l.split()[-1]
      nb_data = nb_data + 1
    # if b"Job_Name" in l:
    if "Job_Name" in l:
      if args.verbose:
        print(l)
      jobname = l.split()[-1]
      nb_data = nb_data + 1
    # if b"job_state" in l:
    if "job_state" in l:
      if args.verbose:
        print(l)
      jobstate = l.split()[-1]
      nb_data = nb_data + 1
    # if b"resources_used.walltime" in l:
    if "resources_used.walltime" in l:
      if args.verbose:
        print(l)
      timeused = l.split()[-1]
      nb_data = nb_data + 1
    # if b"Resource_List.walltime" in l:
    if "Resource_List.walltime" in l:
      if args.verbose:
        print(l)
      timereqd = l.split()[-1]
      nb_data = nb_data + 1

    if nb_data == 5:
      job = {
        "id": jobid,
        "name": jobname,
        "status": jobstate,
        "treqd": timereqd,
        "tused": timeused,
      }
      jobs.append(job)

      nb_data = 0
      jobid = None
      jobname = None
      timeused = None
      jobstate = None
      timereqd = None


#######################################################################
def parse_squeue(proc):

  jobs = []
  job = dict()

  jobid = None
  jobname = None
  jobstate = None
  timeused = None
  timereqd = None
  
  for l in proc.stdout.split(b"\n"):
    if l:
      (jobid, jobname, jobstate, timeused, timereqd) = (
         str(l, "UTF8").strip("'").split()
      )
      if "GW" in jobname:
        job = {
          "id": jobid,
          "name": jobname,
          "status": jobstate,
          "treqd": timereqd,
          "tused": timeused,
        }
        jobs.append(job)

      jobid = None
      jobname = None
      timeused = None
      jobstate = None
      timereqd = None

  # return jobs
  return sorted(jobs, key=lambda d: d["name"]) 


#######################################################################
def get_nb_days(nb_file):

  nb_type = 4
  if (nb_file % 4):
    nb_type = 5
  nb_days = int(nb_file / nb_type)

  return nb_days, nb_type


#######################################################################

if __name__ == "__main__":

  run_deb = dt.datetime.now()
  # freemem()

  args = get_arguments()
  if args.verbose:
    print(args)

  user = os.getlogin()
  if args.verbose:
    print(user)

  dirin = Path("/bdd/CIRS-LMD/ERA5_averages")
  # # dirin = Path("/data/slipsl/GEWEX/ERA5_averages")
  # dirin = Path("/homedata/slipsl/GEWEX/ERA5_averages")
  # # dirin = Path("/bdd/CIRS-LMD/ERA5_averages")
  # # dirin = Path("/home_local/slipsl/GEWEX/output")
  # dirout = Path("/bdd/CIRS-LMD/ERA5_averages")

  if args.param:

    if args.param.runtype == "0":
      runtypes = (1, 2, 3, 4, 5, 6)
    else:
      runtypes = (int(args.param.runtype), )

    if args.param.year.year < 2008:
      years = tuple(
        dt.datetime.strptime(F"{year}", "%Y")
        for year in range(2008, 2023)
      )
    else:
      years = (args.param.year, )


    lensep = 8 + (15 * len(runtypes))
    title = F"| {4*' '} |"
    for runtype in runtypes:
      instru = gwp.InstruParam(runtype)
      title = title + F" {instru.name:7}{3*' '}{instru.ampm:2} |"
    print(F"{lensep*'='}\n{title}\n{lensep*'='}")

    for year in years:
      ligne1 = F"| {year:%Y} |"
      ligne2 = F"| {4*' '} |"

      days_in_year = (
        dt.datetime(year.year + 1, 1, 1) - dt.datetime(year.year, 1, 1)
      ).days

      for runtype in runtypes:
        instru = gwp.InstruParam(runtype)
        pattern = get_pattern(instru, year)

        nb_file = 0
        last_file = None

        p = dirin.joinpath(F"{year:%Y}")
        for dirname in sorted(x for x in p.iterdir() if x.is_dir()):
          if args.verbose:
            print(dirname, len(list(dirname.glob(pattern))))
          nb_file = nb_file + len(list(dirname.glob(pattern)))
          if len(list(dirname.glob(pattern))) > 0:
            last_file = sorted(list(dirname.glob(pattern)))[-1]

        nb_pad = 8
        if last_file is not None:
          last_file = last_file.suffixes[0].lstrip(".")
          nb_pad = 4

        (nb_days, nb_type) = get_nb_days(nb_file)

        if nb_file == 0:
          ligne1 = ligne1 + F" {bcolors.BoldRed}None{bcolors.NoColor}{8*' '} |"
          ligne2 = ligne2 + F" {12*' '} |"
        elif nb_days == days_in_year:
          ligne1 = ligne1 + F" {bcolors.BoldGreen}Done{bcolors.NoColor}{8*' '} |"
          ligne2 = ligne2 + F" {12*' '} |"
        else:
          ligne1 = ligne1 + F" {nb_file:4} = {nb_type:1}x{nb_days:3} |"
          # ligne2 = ligne2 + F" last: {last_file}{nb_pad*' '} |"
          ligne2 = ligne2 + F" {last_file}{nb_pad*' '} |"

      # print(F"{ligne1}\n{ligne2}")
      print(F"{ligne1}")
      if (ligne1.count("Done") < len(runtypes) and
          ligne1.count("None") < len(runtypes)):
        print(F"{ligne2}")
    print(F"{lensep*'='}")

    # print(F"{63*'='}")
    # for runtype in runtypes:
    #   instru = gwp.InstruParam(runtype)
    #   for year in years:
    #     pattern = get_pattern(instru, year)

    #     nb_file = 0
    #     last_file = None

    #     p = dirin.joinpath(F"{year:%Y}")
    #     for dirname in sorted(x for x in p.iterdir() if x.is_dir()):
    #       if args.verbose:
    #         print(dirname, len(list(dirname.glob(pattern))))
    #       nb_file = nb_file + len(list(dirname.glob(pattern)))
    #       if len(list(dirname.glob(pattern))) > 0:
    #         last_file = sorted(list(dirname.glob(pattern)))[-1]

    #     nb_pad = 4
    #     if last_file is not None:
    #       last_file = last_file.suffixes[0].lstrip(".")
    #       nb_pad = 0

    #     # nb_type = 5
    #     # if not (nb_file % 4):
    #     #   nb_type = 4
    #     # nb_days = int(nb_file / nb_type)
    #     (nb_days, nb_type) = get_nb_days(nb_file)

    #     print(
    #       F"| {instru.name:7} {instru.ampm} {year:%Y} : "
    #       F"{nb_file:4} ({nb_type:1}x{nb_days:3}) "
    #       F"input files (last: {last_file})"
    #       F"{nb_pad*' '} |"
    #     )
    #   print(F"{63*'='}")

  spirit = ("spirit", "spiritx")
  if any(h in socket.gethostname() for h in spirit):
    cmd = ["squeue", "-u", user, "-ho", "'%i %j %t %M %l'"]
    fg_spirit = True
  else:
    cmd = ["qstat", "-fu", user]
    fg_spirit = False
  
  proc = subprocess.run(
    cmd,
    check=True,
    stdout=subprocess.PIPE
  )

  jobs = parse_proc(proc, fg_spirit)

  if jobs:
    linelen = 114
    print(
      F"{linelen * '='}\n"
      F"| {'Job ID':9} "
      F"| {'Job Name':18} "
      F"| {'St':2} "
      F"| {'Time':^23} "
      F"| {'Instru':^12} "
      F"| {'Year':4} "
      F"| {'Files nb':^12} "
      F"| {'Last File':9} "
      F"|\n"
      F"{linelen * '='}"
    )
    for job in jobs:
      (instru, year, bornes) = get_params(str(job['name']))

      fg_first = True
      for i in range(bornes[0].year, bornes[1].year + 1):
        if i > 2022:
          continue
        y = dt.datetime.strptime(F"{i}", "%Y")
        # print(i)
        # print(bornes)
        subbounds = (
          max(
            bornes[0],
            dt.datetime(year=i, month=1, day=1)
          ),
          min(
            bornes[1],
            dt.datetime(year=i, month=12, day=31)
          )
        )
        pattern = get_pattern(instru, y)
        # (last_file, nb_file) = get_files(y, bornes, pattern)
        (last_file, nb_file) = get_files(y, subbounds, pattern)
        # print(last_file, nb_file)


        nb_pad = 5
        if last_file is not None:
          last_file = last_file.suffixes[0].lstrip(".")
          nb_pad = 1

          # nb_type = 4
          # if (nb_file % 4):
          #   nb_type = 5
          # # if not (nb_file % 5):
          # #   nb_type = 5
          # nb_days = int(nb_file / nb_type)
          (nb_days, nb_type) = get_nb_days(nb_file)

        sep = "| "
        part2 = (
          F"{sep}{i:4} "
          F"{sep}{nb_file:4} ({nb_type:1}x{nb_days:3}) "
          F"{sep}{last_file}{nb_pad * ' '} "
          F"|"
        )
        if fg_first:
          part1 = (
            F"| {job['id']:9} "
            F"{sep}{job['name']:18} "
            F"{sep}{job['status']:2} "
            F"{sep}{job['tused']:>10} / {job['treqd']:>10} "
            F"{sep}{instru.name:7} "
            F"{sep}{instru.ampm:2} "
          )
        else:
          sep = "  "
          part1 = (
            F"| {'':9} "
            F"{sep}{'':18} "
            F"{sep}{'':2} "
            F"{sep}{'':23} "
            F"{sep}{'':7} "
            F"{sep}{'':2} "
          )

        print(F"{part1}{part2}")


        # print(
        #   F"| {job['id']:9} "
        #   F"| {job['name']:18} "
        #   F"| {job['status']:2} "
        #   F"| {job['tused']:>10} / {job['treqd']:>10} "
        #   F"| {instru.name:7} "
        #   F"| {instru.ampm:} "
        #   F"| {year:%Y} "
        #   F"| {nb_file:4} ({nb_type:1}x{nb_days:3}) "
        #   F"| {last_file}{nb_pad * ' '} "
        #   F"|"
        # )

        fg_first = False

      # pattern = get_pattern(instru, year)
      # (last_file, nb_file) = get_files(year, bornes, pattern)

      # nb_pad = 5
      # if last_file is not None:
      #   last_file = last_file.suffixes[0].lstrip(".")
      #   nb_pad = 1

      # nb_type = 4
      # if not (nb_file % 5):
      #   nb_type = 5
      # nb_days = int(nb_file / nb_type)

      # print(
      #   F"| {job['id']:9} "
      #   F"| {job['name']:18} "
      #   F"| {job['status']:2} "
      #   F"| {job['tused']:>10} / {job['treqd']:>10} "
      #   F"| {instru.name:7} "
      #   F"| {instru.ampm:} "
      #   F"| {year:%Y} "
      #   F"| {nb_file:4} ({nb_type:1}x{nb_days:3}) "
      #   F"| {last_file}{nb_pad * ' '} "
      #   F"|"
      # )

    print( F"{linelen * '='}")

  # print(F"\n{72*'='}\nRun ended in {dt.datetime.now() - run_deb}")


  exit()

