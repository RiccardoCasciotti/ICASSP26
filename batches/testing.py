import json
import os
import subprocess
import uuid
import shutil
import torch
import json
import os
import shlex
import subprocess
import random
import numpy as np
TEST = False # we reduced the epochs, reduced the folds, reduced the tasks, reduced the layers to 4
SHMH = False
SINGLE = False

n_experiments = 2



dataset="ESC50" # options: URBANSOUND8K, ESC50

if dataset == "ESC50":
    if SINGLE:
        n_tasks = 1
    classes_per_task = 5 # fixed to 30 + 5 + 5 + 5 + 5
    n_tasks = 5

elif dataset == "URBANSOUND8K":
    classes_per_task = 2
    n_tasks = 5

evaluated_tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if TEST: 
    n_experiments = 1
    n_tasks = 2

id = "_full_run_gitcode"
folder_id = f"_{id}{n_tasks}tasks"
parent_f_id = f"experiments/EXP_{dataset}_{classes_per_task}C"



cl_hyper = {
                    'training_mode': 'consecutive',
                    'top_k': 0.6,
                    'topk_lock': False,
                    'high_lr': 0.15,
                    'low_lr': 0.9,
                    't_criteria': 'activations',
                    'delta_w_interval': 5,
                    'heads_basis_t': 0.90,
                    'n_tasks': n_tasks, 
                    'classes_per_task': classes_per_task
                }

def folder_check(path):
    print(os.path.exists(f"{BASE_PATH}/" + path))
    print(f"{BASE_PATH}" + path)
    return os.path.isdir("{BASE_PATH}/" + path)
def execute_bash_command(evaluated_tasks: list, n_tasks: int, command: str, classes=[]):
    
    if TEST:
        sols = [(True, True)]
    else: 
        sols = [(True, True), (False, True)]
    if dataset == "ESC50":
        if SINGLE:
            sols = [(False, False)]
        else:
            sols = [(False, True), (True, True)]

    cl_hyper["SINGLE"] = SINGLE

    f = open("/projappl/project_462001198/casciott/ICASSP26/batches/configs.sh", "w")
    f.write("#!/bin/bash\n")
    
    for sol in sols:
        cl_hyper['cf_sol'] = sol[0]
        cl_hyper['head_sol'] = sol[1]
        cl_hyper['classes_per_task'] = classes_per_task
        
        
        for sc in classes: 
        
            selected_classes_str = shlex.quote(json.dumps(sc))
            evaluated_tasks_str = shlex.quote(json.dumps(evaluated_tasks))
            
            command1 = (
                command +
                f"{cl_hyper['training_mode']} "
                f"{cl_hyper['cf_sol']} "
                f"{cl_hyper['head_sol']} "
                f"{cl_hyper['top_k']} "
                f"{cl_hyper['high_lr']} "
                f"{cl_hyper['low_lr']} "
                f"{cl_hyper['t_criteria']} "
                f"{cl_hyper['delta_w_interval']} "
                f"{cl_hyper['heads_basis_t']} "
                f"{selected_classes_str} "
                f"{cl_hyper['n_tasks']} "
                f"{evaluated_tasks_str} "
                f"{cl_hyper['classes_per_task']} "
                f"{cl_hyper['topk_lock']} "
                f"{folder_id} "
                f"{parent_f_id} "
                f"{SHMH}"

            )
                
        
            f.write(f'{command1}\n')
                   
        
        if TEST:
            print("!!!! WARNING: BREAK OPERATION IS ON IN TESTING")
            break
            
    f.close() 



BASE_PATH="/scratch/project_462001198/casciott"

if not os.path.isdir(f"{BASE_PATH}/experiments"):
    os.mkdir(f"{BASE_PATH}/experiments")
if not os.path.isdir(f"{BASE_PATH}/{parent_f_id}"):
    os.mkdir(f"{BASE_PATH}/{parent_f_id}")
            

command = f'cd /projappl/project_462001198/casciott/ICASSP26/batches/classes_CL/continual_learning/ && sbatch {dataset}.sh ' 


if dataset == "ESC50":  
    all_classes = list(range(50))
    all_classes_ordered = all_classes.copy()
elif dataset == "URBANSOUND8K":
    all_classes = list(range(10))
    all_classes_ordered = all_classes.copy()
classes = []

if dataset == "ESC50": 
    for i in range(n_experiments):
        task_classes = []
        random.shuffle(all_classes)
        task_classes.append(all_classes[:30])
        for i in range(30, 50, 5):
            task_classes.append(all_classes[i:i+5])
        classes.append(task_classes)
    final = classes
elif dataset == "URBANSOUND8K":
    for i in range(n_experiments):
        task_classes = []
        random.shuffle(all_classes)
        task_classes.append(all_classes[:classes_per_task])
        for i in range(classes_per_task, n_tasks*classes_per_task, classes_per_task):
            task_classes.append(all_classes[i:i+classes_per_task])
        classes.append(task_classes)
    final = classes
print(n_experiments)
if SINGLE:
    final = [[all_classes_ordered]]
print("Classes selected for each task in each experiment: ", final)

if folder_check(f"{parent_f_id}/TASKS_CL_{dataset +  folder_id}"):
    res = input(f"!!!! WARNING A FOLDER NAMED 'TASKS_CL_{ dataset +  folder_id}' already exits, press Y to continue anyways or N to abort: ")
    if res == "Y":
        execute_bash_command(evaluated_tasks, n_tasks, command, final)
else:
    execute_bash_command(evaluated_tasks, n_tasks, command, final)



