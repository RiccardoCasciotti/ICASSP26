#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --account=project_462001198
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00

module load LUMI/22.08
module load singularity

singularity exec --rocm \
  --env HIP_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES} \
  --bind /projappl/project_462001198/casciott/ICASSP26 \
  --pwd /projappl/project_462001198/casciott/ICASSP26 \
  /projappl/project_462001198/casciott/ICASSP26/softhebb_env/softhebb.sif \
  bash -c "unset ROCR_VISIBLE_DEVICES; python3 ray_test.py"
