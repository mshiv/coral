#!/bin/bash
#SBATCH --job-name=lisflood-sav
#SBATCH --account=gts-arobel3-atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=12G
#SBATCH --output=out.lisflood-sav.%j
#SBATCH --error=err.lisflood-sav.%j
#SBATCH --time=08:00:00

# LISFLOOD-FP is OpenMP-only (no MPI) -> single node, threaded.
# Use the bug-fixed release binary built with config/pace-gcc-cpu.

module load gcc/12.3.0 netcdf-c/4.9.2-gszew36

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -s unlimited

cd "$SLURM_SUBMIT_DIR"
/path/to/LISFLOOD-FP-trunk/lisflood savannah.par
