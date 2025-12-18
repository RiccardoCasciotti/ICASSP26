#!/usr/bin/env bash
#
# A LUMI SLURM batch script for the LUMI PyTorch single GPU test example from
# https://github.com/DeiC-HPC/cotainr
#
#SBATCH --job-name=singlegpu_gputest_example
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output=output.txt
#SBATCH --error=job.err 
#SBATCH --partition=small-g
#SBATCH --time=00:05:00
#SBATCH --account=project_462001198

srun singularity exec \
    --bind /project/project_462001198/casciott/ICASSP26 \
    --pwd  /project/project_462001198/casciott/ICASSP26 \
    lumi_pytorch_rocm_demo.sif \
    python conda_pytorch_rocm/pytorch_singlegpu_gputest.py
