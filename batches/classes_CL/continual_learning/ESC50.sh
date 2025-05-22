#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --account=project_462000765     # account name
#SBATCH --gpus=1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00               # time limits: 1/2 hour
#SBATCH --mem=128G
#SBATCH --error=ESC50/job.err            # standard error file
#SBATCH --output=ESC50/job.out           # standard output file
module load CrayEnv
module load cotainr
module load rocm
# srun python3 /projappl/project_462000765/casciott/DCASE25/SoftHebb-main/continual_learning.py --preset 6SoftHebbCnnESC --resume all --model-name 'ESC50_CL' --dataset-unsup ESC50_1 --dataset-sup ESC50_50 --continual_learning True --evaluate True --training-mode $1 --cf-sol $2 --head-sol $3 --top-k $4 --high-lr $5 --low-lr $6 --t-criteria $7 --delta-w-interval $8 --heads-basis-t $9 --selected-classes "${10}" --n-tasks "${11}" --evaluated-task "${12}" --classes-per-task "${13}" --topk-lock "${14}" --folder-id "${15}" --parent-f-id "${16}" 
srun singularity exec --rocm \
    --bind /projappl/project_462000765/casciott/DCASE25 \
    --pwd  /projappl/project_462000765/casciott/DCASE25 \
    /projappl/project_462000765/casciott/DCASE25/softhebb_env/softhebb.sif \
     python3 /projappl/project_462000765/casciott/DCASE25/SoftHebb-main/continual_learning.py --preset 6SoftHebbCnnESC --resume all --model-name 'ESC50_CL' --dataset-unsup ESC50_1 --dataset-sup ESC50_50 --continual_learning True --evaluate True --training-mode $1 --cf-sol $2 --head-sol $3 --top-k $4 --high-lr $5 --low-lr $6 --t-criteria $7 --delta-w-interval $8 --heads-basis-t $9 --selected-classes "${10}" --n-tasks "${11}" --evaluated-task "${12}" --classes-per-task "${13}" --topk-lock "${14}" --folder-id "${15}" --parent-f-id "${16}"

# srun --partition=standard-g --account=project_462000765 singularity exec --rocm --bind /projappl/project_462000765/casciott/DCASE25 --pwd  /projappl/project_462000765/casciott/DCASE25 /projappl/project_462000765/casciott/DCASE25/softhebb_env/softhebb.sif pip list
# srun singularity exec --rocm \
#     --bind /projappl/project_462000765/casciott/DCASE25 \
#     --pwd  /projappl/project_462000765/casciott/DCASE25 \
#     /projappl/project_462000765/casciott/DCASE25/softhebb_env/softhebb.sif \
#      python3 /projappl/project_462000765/casciott/DCASE25/batches/dummy.py