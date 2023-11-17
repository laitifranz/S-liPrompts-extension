#!/bin/bash
## THIS IS A UNIQUE LAUNCHER FOR SBATCH

############################################
# Parameters. Reference https://slurm.schedmd.com/sbatch.html
RUN_NAME="seed_2_labelsmooth_0_1_patience_10"
SCRIPT_TO_RUN="train.sh"

GPUS_PER_NODE=1
PROJECT_FOLDER="S-Prompts"
VIRTUAL_ENV="torch-cuda12.1-py3.9"
############################################


# Derived paths
RUN_DIR="/home/francesco.laiti/workspace/${PROJECT_FOLDER}/logs/${RUN_NAME}" # _$(date +%Y%m%d_%H%M%S)
LOG_DIR="${RUN_DIR}"
OUTPUT_FILE="${LOG_DIR}/output_%j.log"
ERROR_FILE="${LOG_DIR}/error_%j.log"
SCRIPT_FILE="launchers/${SCRIPT_TO_RUN}"

# Ask job name confirm
read -p "-- Set job name ${RUN_NAME}, proceed? (y/n) : " answer
case $answer in
    [nN] | [nN][oO])
        exit -1
        ;;
esac

# Create directories
mkdir -p "${RUN_DIR}" || { echo "Error: Failed to create ${RUN_DIR}"; exit 1; }
mkdir -p "${LOG_DIR}" || { echo "Error: Failed to create ${LOG_DIR}"; exit 1; }

echo "-- Launching $SCRIPT_FILE for project $PROJECT_FOLDER with name $RUN_NAME --"

# Launch job on cluster
sbatch  --gres=gpu:1g.6gb:${GPUS_PER_NODE} \
        --job-name=${RUN_NAME} \
        --chdir=/home/francesco.laiti/workspace/${PROJECT_FOLDER} \
        --output=${OUTPUT_FILE} \
        --error=${ERROR_FILE} \
        --export=VENV_PATH=/home/francesco.laiti/venvs/${VIRTUAL_ENV}/bin \
        ${SCRIPT_FILE}