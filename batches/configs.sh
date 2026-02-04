#!/bin/bash
cd /projappl/project_462001198/casciott/ICASSP26/batches/classes_CL/continual_learning/ && sbatch URBANSOUND8K.sh consecutive True True 0.6 0.15 0.9 activations 5 0.9 '[[5, 9], [4, 1], [8, 2], [3, 7], [6, 0]]' 5 '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]' 2 False __full_run_testing5tasks experiments/EXP_URBANSOUND8K_2C False
