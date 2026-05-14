#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --account=project_462001198     # account name
#SBATCH --gpus=1
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00               # time limits: 1/2 hour
#SBATCH --mem=128G
#SBATCH --error=ESC50/job.err            # standard error file
#SBATCH --output=ESC50/job.out           # standard output file

module load CrayEnv
module load cotainr
module load rocm
srun singularity exec \
                 --rocm \
    --bind /projappl/project_462001198/casciott/ICASSP26 \
    --bind /scratch/project_462001198/casciott \
    --pwd  /projappl/project_462001198/casciott/ICASSP26 \
    /scratch/project_462001198/casciott/softhebb_env/softhebb.sif \
    python3 /projappl/project_462001198/casciott/ICASSP26/SoftHebb-main/continual_learning.py --preset "${18}" --resume all --model-name 'ESC50_CL' --dataset-unsup ESC50_1 --dataset-sup ESC50_100 --continual_learning True --evaluate True --training-mode $1 --cf-sol $2 --head-sol $3 --top-k $4 --high-lr $5 --low-lr $6 --t-criteria $7 --delta-w-interval $8 --selected-classes "$9" --n-tasks "${10}" --evaluated-task "${11}" --classes-per-task "${12}" --topk-lock "${13}" --folder-id "${14}" --parent-f-id "${15}" --presets_path "${16}" --datasets_path "${17}" --results_path "${19}"
