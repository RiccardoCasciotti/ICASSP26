#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --account=project_462000765     # account name
#SBATCH --gpus-per-task=2
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00               # time limits: 1/2 hour
#SBATCH --error=TEST/job.err            # standard error file
#SBATCH --output=TEST/job.out           # standard output file
module load CrayEnv
module load cotainr

# srun singularity exec \
#     --bind /project/project_462000765/casciott/DCASE25 \
#     --pwd  /project/project_462000765/casciott/DCASE25 \
#     /project/project_462000765/casciott/DCASE25/softhebb_env/softhebb.sif \
#      python3 /project/project_462000765/casciott/DCASE25/batches/test.py
srun singularity exec \
    --bind /projappl/project_462000765/casciott/DCASE25 \
    --pwd  /projappl/project_462000765/casciott/DCASE25 \
    /projappl/project_462000765/casciott/DCASE25/softhebb_env/softhebb.sif \
     python3 /projappl/project_462000765/casciott/DCASE25/batches/test.py