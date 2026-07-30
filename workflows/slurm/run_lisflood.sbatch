#!/bin/bash
#SBATCH --job-name=lisflood-sav
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=out.lisflood-sav.%j
#SBATCH --error=err.lisflood-sav.%j
#SBATCH --time=08:00:00

# LISFLOOD-FP is OpenMP-only (no MPI) -> single node, threaded.
# Use the bug-fixed release binary built with config/pace-gcc-cpu.

module load gcc/12.3.0 netcdf-c/4.9.2-gszew36

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -s unlimited

cd "$SLURM_SUBMIT_DIR"
# savannah.par enables netcdf_out, which segfaults on finalisation in this build, and
# 00_setup.sbatch deletes it. Run the ASCII par.
/storage/home/hcoda1/6/smurugan9/data/software/LISFLOOD-FP-trunk/lisflood savannah_ascii.par
