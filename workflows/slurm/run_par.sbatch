#!/bin/bash
#SBATCH --job-name=lisflood
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=out.lisflood.%j
#SBATCH --error=err.lisflood.%j
#SBATCH --time=08:00:00

# Run one LISFLOOD par file.
#   sbatch run_par.sbatch <run_dir> <par_file>
# run_lisflood.sbatch names a single par, so each run needed its own copy of the script.
#
# LISFLOOD-FP is OpenMP-only, so this is one node with threads. Thread scaling saturates
# at 16 on this hardware. Raise --time for 2 m runs: they cost roughly 10x the 4 m runs,
# because the cell count is 4x and the CFL limit dt <= a*dx/sqrt(g*h) scales with dx.

module load gcc/12.3.0 netcdf-c/4.9.2-gszew36
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -s unlimited

cd "$1" || exit 1
/storage/home/hcoda1/6/smurugan9/data/software/LISFLOOD-FP-trunk/lisflood "$2"
