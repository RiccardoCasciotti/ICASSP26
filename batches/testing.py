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
TEST = True # we reduced the epochs, reduced the folds, reduced the tasks, reduced the layers to 4
SHMH = False
SINGLE = False

classes_per_task = 2
n_experiments = 1
n_tasks = 5

evaluated_tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]
#neuromodAI/SoftHebb-main/experiments/EXP_C100_4C/TASKS_CL_CIFAR100_d3_6tasks
#neuromodAI/SoftHebb-main/experiments/EXP_C100_2C/TASKS_CL_CIFAR100_c1_big_6tasks

    

data_num = 1 # set to 2 to use in multi dataset CL mode, otherwise to 1 for tasks from the same dataset.
dataset="ESC50"
dataset2 = "C10"

if dataset == "ESC50":
    classes_per_task = 50
    n_tasks = 5
    if SINGLE:
        n_tasks = 1
elif dataset == "URBANSOUND8K":
    
    classes_per_task = 10
    n_tasks = 5
    if SINGLE:
        n_tasks = 1

if TEST: 
    n_experiments = 1
    n_tasks = 2

id = "_test_run_"
folder_id = f"_{id}{n_tasks}tasks"


if data_num == 1:
    parent_f_id = f"experiments/EXP_{dataset}_{classes_per_task}C"
else:
    parent_f_id = f"experiments/EXP_{dataset}_{dataset2}"


# C100, C10, STL10, IMG, ESC50

cl_hyper = {
                    'training_mode': 'consecutive',
                    'top_k': 0.6,
                    'topk_lock': False,
                    'high_lr': 0.15,
                    'low_lr': 0.9,
                    't_criteria': 'activations', # KSE or activations
                    'delta_w_interval': 5,
                    'heads_basis_t': 0.90,
                    'n_tasks': n_tasks, 
                    'classes_per_task': classes_per_task
                }

# for root, dirs, files in os.walk("/leonardo_work/IscrC_CATASTRO/rcasciot/neuromodAI/SoftHebb-main/experiments/EXP_C10_2C/TASKS_CL_CIFAR10_a1_8tasks", topdown=False):
#         for file in files:
#             if ".json" not in file:
#                 continue
#             with open(os.path.join(root, file), "r") as f:
#                 json_obj = json.load(f)
                
                
#                # print(dataset, json_obj["R0"]['dataset_sup']["name"])
#                 if "b4" not in list(json_obj["model_config"].keys()): 
#                     result = subprocess.run(f"rm -rf /leonardo_work/IscrC_CATASTRO/rcasciot/neuromodAI/SoftHebb-main/experiments/EXP_C10_2C/TASKS_CL_CIFAR10_a1_8tasks/{file}", shell=True, capture_output=False, text=True)
#                     print(result.stdout)
#                     if result.stderr:
#                         print("Error:", result.stderr)
                    


def folder_check(path):
    print(os.path.exists(f"{BASE_PATH}/" + path))
    print(f"{BASE_PATH}" + path)
    return os.path.isdir("{BASE_PATH}/" + path)
def execute_bash_command(evaluated_tasks: list, n_tasks: int, command: str, classes=[]):
    modes = ["successive", "consecutive", "simultaneous"]
    lrs = [(0.0, 1.0), (2000, 1.0), (0.2, 0.8)]
    if TEST:
        sols = [(True, True)]
    else: 
        sols = [(True, True), (False, True)]
    if dataset == "ESC50":
        if SINGLE:
            sols = [(False, False)]
        else:
            sols = [(False, True), (True, True)]
    topks = [0.1, 0.2, 0.5, 0.7, 0.85, 0.9, 1.0]
    delta_w_intervals = [20, 100, 300]
    lr = lrs[2]
    mode = modes[1]
    cl_hyper["SINGLE"] = SINGLE
    
    for sol in sols:
        cl_hyper['cf_sol'] = sol[0]
        cl_hyper['head_sol'] = sol[1]
        cl_hyper['classes_per_task'] = classes_per_task
        
        if data_num == 1: 
            for sc in classes: # this corresponds to how many experiments we want to run
            
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
                    f"{SHMH}" + '"'

                )
                  
                result = subprocess.check_output(
                    command1,
                    shell=True,
                    cwd="/projappl/project_462001198/casciott/ICASSP26/batches/classes_CL/continual_learning/",
                )
              
                # result = subprocess.run(command1, shell=False, capture_output=False, text=True, )
                print("out: ", result)
                
        
        if TEST:
            print("!!!! WARNING: BREAK OPERATION IS ON IN TESTING")
            break
            



# command = f"rm -rf -d /leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/Training/results/hebb/result/network && mkdir /leonardo_work/{USER}/rcasciot/neuromodAI/SoftHebb-main/Training/results/hebb/result/network"
# result = subprocess.run(command, shell=True, capture_output=False, text=True)
    
# print(result.stdout)
# if result.stderr:
#     print("Error:", result.stderr)

   

         # Apple Silicon GPU
BASE_PATH="/scratch/project_462001198/casciott"

if not os.path.isdir(f"{BASE_PATH}/experiments"):
    os.mkdir(f"{BASE_PATH}/experiments")
if not os.path.isdir(f"{BASE_PATH}/{parent_f_id}"):
    os.mkdir(f"{BASE_PATH}/{parent_f_id}")
            

if data_num == 1: 
    command = f'bash -lc "source /etc/profile.d/modules.sh && module load slurm && sbatch {dataset}.sh' 

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
            task_classes.append(all_classes[:2])
            for i in range(2, 10, 2):
                task_classes.append(all_classes[i:i+2])
            classes.append(task_classes)
        final = classes
    print(n_experiments)
    if SINGLE:
        final = [[all_classes_ordered]]
        #  final = [[[21, 31, 2, 20, 34, 22, 16, 43, 42, 40, 45, 36, 33, 1, 12, 24, 28, 15, 26, 9, 44, 27, 32, 6, 47, 19, 5, 4, 46, 18], [8, 35, 29, 48, 13], [3, 0, 25, 7, 30], [49, 11, 14, 23, 37], [17, 10, 41, 38, 39]]]
    print("final: ", final)
   
    if dataset == "ESC50": 
        dataset1 = "ESC50"
    elif dataset == "URBANSOUND8K":
        dataset1 = "URBANSOUND8K"
    if folder_check(f"{parent_f_id}/TASKS_CL_{dataset1 +  folder_id}"):
        res = input(f"!!!! WARNING A FOLDER NAMED 'TASKS_CL_{ dataset +  folder_id}' already exits, press Y to continue anyways or N to abort: ")
        if res == "Y":
            execute_bash_command(evaluated_tasks, n_tasks, command, final)
    else:
        execute_bash_command(evaluated_tasks, n_tasks, command, final)



