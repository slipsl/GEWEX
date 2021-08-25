#!/bin/sh

#PBS -N GWX_TEST
#PBS -q std
#PBS -o main2.out
#PBS -e main2.err
##PBS -l mem=15gb,vmem=15gb

module purge
module load python/3.6-anaconda50
source activate gewex2

cd $PBS_O_WORKDIR

python src/gewex_main2.py -f 1 20080220 20080220