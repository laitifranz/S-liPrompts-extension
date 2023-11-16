#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8192M
#SBATCH --time=5:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.laiti@studenti.unitn.it
#SBATCH --nodes=1
#SBATCH --partition=edu-thesis
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# Log info
echo "-- Job name: $SLURM_JOB_NAME --"
echo "-- Directory: $SLURM_SUBMIT_DIR --"
echo "-- Env: $VENV_PATH --"
echo "-- Date: $(date +%Y%m%d_%H%M%S) --"
echo ""

# Activate Virtual Environment
source $VENV_PATH/activate

# Run script, consider that you are in the main folder of the project declared in launcher.sh
srun $VENV_PATH/python main.py # --config configs/cddb_slip.json


## trash ##
# module load cuda/12.1
# export CUDA_VISIBLE_DEVICES=MIG-ea075e70-2bd0-584b-8640-251c07f07b41,MIG-672be968-4fe2-54e1-9ff3-3dd358a04a43