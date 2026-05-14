
import argparse
import ast
import gc
import itertools
import os
import subprocess
import sys
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as op
import json
from utils import load_presets, get_device, load_config_dataset, seed_init_fn, str2bool
from model import load_layers
from train import run_sup, run_unsup, check_dimension, training_config, run_hybrid
from log_m import Log
import warnings
import copy

from utils import CustomStepLR, double_factorial
from engine_cl import evaluate_sup, train_sup, train_unsup, evaluate_unsup, getActivation, evaluate_sup_multihead
from dataset import make_data_loaders
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np 


BASE_PATH="/scratch/project_462001198/casciott"

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Multi layer Hebbian Training Continual Learning  implementation')

parser.add_argument('--continual_learning', choices=[True, False], default=False,
                    type=str2bool)

parser.add_argument('--results_path', default=None,
                    type=str )
parser.add_argument('--preset', default=None,
                    type=str, help='Preset of hyper-parameters ' +
                                   ' | ' +
                                   ' (default: None)')
parser.add_argument('--folder-id', default=None,
                    type=str )
parser.add_argument('--parent-f-id', default=None,
                    type=str )


parser.add_argument('--datasets_path', default=None,
                    type=str )

parser.add_argument('--presets_path', default=None,
                    type=str )

parser.add_argument('--dataset-unsup-1', default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | ' +
                                   ' (default: None)')

parser.add_argument('--dataset-sup-1', default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | ' +
                                   ' (default: None)')

parser.add_argument('--training-mode', choices=['successive', 'consecutive', 'simultaneous'], default='consecuttive',   ###################
                    type=str, help='Training possibilities ' +
                                   ' | '.join(['successive', 'consecutive', 'simultaneous']) +
                                   ' (default: successive)')

parser.add_argument('--resume', choices=[None, "all", "without_classifier"], default=None,
                    type=str, help='Resume Model ' +
                                   ' | '.join(["best", "last"]) +
                                   ' (default: None)')

parser.add_argument('--model-name', default=None, type=str, help='Model Name')

parser.add_argument('--training-blocks', default=None, nargs='+', type=int,
                    help='Selection of the blocks that will be trained')

parser.add_argument('--seed', default=None, type=int,
                    help='')

parser.add_argument('--gpu-id', default=0, type=int, metavar='N',
                    help='Id of gpu selected for training (default: 0)')

parser.add_argument('--save', default=True, type=str2bool, metavar='N',
                    help='')

parser.add_argument('--validation', default=False, type=str2bool, metavar='N',
                    help='')

parser.add_argument('--evaluate', default=False, type=str2bool, metavar='N',
                    help='')
parser.add_argument('--topk-lock', default=False, type=str2bool, metavar='N',
                    help='')
parser.add_argument('--skip-1', default=False, type=str2bool, metavar='N',
                    help='Set to True if you want to skip the training on the first dataset and directly retrieve a model to train it again on the second dataset (you don \'t have to specify a preset if set True) ')
parser.add_argument('--classes-per-task', default=-1, type=int,
                    help='The continual learning is organized in tasks made up of different classes of the same dataset. Number of classes belonging to each task.')
parser.add_argument('--dataset-unsup',  default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | '+
                                   ' (default: None)')

parser.add_argument('--dataset-sup',  default=None,
                    type=str, help='Dataset possibilities ' +
                                   ' | ' +
                                   ' (default: None)')

parser.add_argument('--head-sol', choices=[True, False], default='True',   ###################
                    type=str2bool, help='whether continual learning solution is on or off on linear layers' +
                                   ' | '.join(['on', 'off']) +
                                   ' (default: off)')

parser.add_argument('--cf-sol', default="True",   ###################
                    type=str2bool)
parser.add_argument('--top-k', default=0.8,   ###################
                    type=float)
parser.add_argument('--high-lr', default=0.2,   ###################
                    type=float)
parser.add_argument('--low-lr', default=0.8,   ###################
                    type=float)
parser.add_argument('--delta-w-interval', default=100,   ###################
                    type=float)
parser.add_argument('--t-criteria', default="mean",   ###################
                    type=str)
parser.add_argument('--heads-basis-t', default=0.6,   ###################
                    type=float)
parser.add_argument('--selected-classes', default="[[0,3],[5,7]]",   ###################
                    type=str)
parser.add_argument('--n-tasks', default=2,   ###################
                    type=int)
parser.add_argument('--evaluated-tasks', default="[0,1]",   ###################
                    type=str)

# we need first to pass both the datasets, the evaluation parameter is not needed, or it could be if we decide to validate just one model on one dataset. 
# after we passed both the datasets, train the model on the 1st dataset ( the resume all flag must be artificially set to false) and retrieved the model saved. The continual learning flag will cut the dataset, but it must be applied only 
# during the second training of the model. And so the evaluate must be set to true in the last iteration and continual learning again to false.

results = {"count": 0}


def main(blocks, name_model, resume, save, dataset_sup_config, dataset_unsup_config, train_config, gpu_id, evaluate, results, cl_hyper, task_num = -1, dataset_path=None, result_path=None):
    device = get_device()

    model = load_layers(blocks, name_model, resume, dataset_sup_config=dataset_sup_config, batch_size=list(train_config.values())[-1]["batch_size"], cl_hyper=cl_hyper, task_num=task_num, eval=dataset_sup_config["eval"], result_path=result_path)
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.task_num = task_num
    model.joint = dataset_sup_config["joint"]
    model.name_model = name_model
    model.fold_num = dataset_sup_config["fold"]
    depth = 0
    if model.joint:
        
        model.cl_hyper["cf_sol"] = False
    # print("model.heads: ", model.heads)
    # here we obtain the activations of all the layers (which are convolutional layers)
    handles = []

    if not evaluate and model.cl_hyper["cf_sol"]:
        for layer in model.children():
        # check for convolutional layer
            for subl in layer.children():
                        
                for subsubl in subl.children():
                        
                    if subsubl._get_name().__eq__("HebbSoftKrotovConv2d"):
                        handles.append(subsubl.register_forward_hook(getActivation("conv"+str(depth))))
                    if subsubl._get_name().__eq__("Linear"):
                        handles.append(subsubl.register_forward_hook(getActivation("linear"+str(depth))))
                depth += 1
        
        
    
    log = Log(train_config)
    test_loss = 0
    test_acc = 0
    dataset_sup_config["SINGLE"] = cl_hyper["SINGLE"]
    for id, config in train_config.items():
        # if model.joint:
        #     config["batch_size"] = 4
        train_loader, val_loader, test_loader, classes_offset = make_data_loaders(dataset_config=dataset_sup_config, batch_size=config['batch_size'], dataset_path=dataset_path, device=device)
        model.classes_offset = classes_offset

        if evaluate:
            
            if config['mode'] == 'supervised' or config['mode'] == 'hybrid': ## WATCH OUT EVAL LOGGING WORKS ONLY WITH 1 SUPERVISED LAYER
                criterion = nn.CrossEntropyLoss()
                result = {}
                task_cf = None
                if cl_hyper["head_sol"] and not model.joint:
                    res = evaluate_sup_multihead(model, criterion, test_loader, device, return_confusion_matrix=True, result_path=result_path) ##################################
                    test_loss = res[0]
                    test_acc = res[1]
                    if len(res) == 3:
                        task_cf = res[2]
                else: 
                    test_loss, test_acc= evaluate_sup(model, criterion, test_loader, device)
                    # cm, test_loss, test_acc= evaluate_sup(model, criterion, test_loader, device)

                    # plot_confusion_matrix(cm, path=f"{params.parent_f_id}/TASKS_CL_{params.dataset_sup.split('_')[0] +  folder_id}", name=name_model)

                print(f'Accuracy of the network on the task: {test_acc:.3f} %')
                print(f'Test loss on the task: {test_loss:.3f}')

                conv, R1 = model.convergence()
                if type(test_loss) ==  torch.Tensor:
                    metrics = {"test_loss":test_loss.item(), "test_acc": test_acc.item(), "convergence":conv, "R1":R1}
                else: 
                    metrics = {"test_loss":test_loss, "test_acc": test_acc, "convergence":conv, "R1":R1}
                if "dataset_sup" not in results.keys() and "cl_hyper" not in results.keys():
                    metrics["dataset_sup"] = dataset_sup_config.copy()
                    metrics["dataset_unsup"] = dataset_unsup_config.copy()
                    results["cl_hyper"] = cl_hyper
                    
                
                results["FOLD_#"+str(dataset_sup_config["fold"])][f"eval_{task_num}"] = metrics.copy()
                
                results["count"] += 1

                # ACCURACY MATRIX
                if not dataset_sup_config["joint"] :
                    if task_num not in results["accuracy_matrix"]:
                        results["accuracy_matrix"][task_num] = {}
                    if f'FOLD_#{dataset_sup_config["fold"]}' not in results["accuracy_matrix"][task_num]:
                        results["accuracy_matrix"][task_num][f"FOLD_#{dataset_sup_config["fold"]}"] = []
                    if len(results["accuracy_matrix"][task_num][f"FOLD_#{dataset_sup_config["fold"]}"]) < cl_hyper["n_tasks"]:
                        results["accuracy_matrix"][task_num][f"FOLD_#{dataset_sup_config["fold"]}"].append(test_acc.item())

                ###########################

                # JOINT
                if dataset_sup_config["joint"]:
                    if f"FOLD_#{dataset_sup_config["fold"]}" not in results["joint"]:
                        results["joint"][f"FOLD_#{dataset_sup_config["fold"]}"] = []
                    results["joint"][f"FOLD_#{dataset_sup_config["fold"]}"].append(test_acc.item())
                ##############

                # CONFUSION MATRIX

                if task_cf is not None and not model.joint:
                    if f"FOLD_#{dataset_sup_config["fold"]}" not in results["confusion_matrix"]:
                        results["confusion_matrix"][f"FOLD_#{dataset_sup_config["fold"]}"] = {}
                    if f"T{task_num}" not in results["confusion_matrix"][f"FOLD_#{dataset_sup_config["fold"]}"]:
                        results["confusion_matrix"][f"FOLD_#{dataset_sup_config["fold"]}"][f"T{task_num}"] = {}
                    results["confusion_matrix"][f"FOLD_#{dataset_sup_config["fold"]}"][f"T{task_num}"] = task_cf
                #############

        else:
            if config['mode'] == 'unsupervised':
                run_unsup(
                    config['nb_epoch'],
                    config['print_freq'],
                    config['batch_size'],
                    name_model,
                    dataset_unsup_config,
                    model,
                    device,
                    log.unsup[id],
                    blocks=config['blocks'],
                    save=save, 
                    train_loader=train_loader,
                    val_loader=val_loader, 
                    result_path=result_path
                )
                
            elif config['mode'] == 'supervised':
                result = run_sup(
                    config['nb_epoch'],
                    config['print_freq'],
                    config['batch_size'],
                    config['lr'],
                    name_model,
                    dataset_sup_config,
                    model,
                    device,
                    log.sup[id],
                    blocks=config['blocks'],
                    save=save,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    task_num=task_num ,
                    result_path=result_path
                )
                if not dataset_sup_config["joint"]:
                    result["dataset_sup"] = dataset_sup_config.copy()
                    result["dataset_unsup"] = dataset_unsup_config.copy()
                    result["train_config"] = train_config.copy()
                    print("RESULT: ", result)
                    results["FOLD_#"+str(dataset_sup_config["fold"])]["T" + str(task_num)] = result.copy()
                    
                    results["count"] += 1

    if "model_config" not in results.keys():
        results["model_config"] = blocks
    print("first heads: ", len(model.heads))
    for h in handles:
        h.remove()
    handles.clear()

      
def plot_confusion_matrix(cm, path, name, class_names=None, normalize=False, title="Confusion Matrix"):

    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Greys',  # Classic gradient (light to dark)
                cbar=True, 
                xticklabels=class_names, 
                yticklabels=class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    if not os.path.exists(f"{BASE_PATH}/SoftHebb-main/{path}"):
         
        os.mkdir(f"{BASE_PATH}/SoftHebb-main/{path}")
    file = f"{BASE_PATH}/SoftHebb-main/{path}" + "/"+ name + "_CF.png"
    plt.savefig(file)


def procedure(params, name_model, blocks, dataset_sup_config, dataset_unsup_config, evaluate, results, task_num):
    # 
    if params.seed is not None:
        dataset_sup_config['seed'] = params.seed
        dataset_unsup_config['seed'] = params.seed

    if dataset_sup_config['seed'] is not None:
        seed_init_fn(dataset_sup_config['seed'])

    blocks = check_dimension(blocks, dataset_sup_config)

    train_config = training_config(blocks, dataset_sup_config, dataset_unsup_config, params.training_mode,
                                   params.training_blocks)
    
    main(blocks, name_model, params.resume, params.save, dataset_sup_config, dataset_unsup_config, train_config,
          params.gpu_id, evaluate, results, cl_hyper=params.cl_hyper, task_num=task_num, dataset_path=params.datasets_path, result_path=params.results_path)


def save_results(results, path, name):
    print("results: ", results)

    if not os.path.exists(f"{BASE_PATH}/{path}"):
         
        os.mkdir(f"{BASE_PATH}/{path}")
    file = f"{BASE_PATH}/{path}" + "/"+ name + ".json"
     
    with open(file, 'w') as f:
        json.dump(results, f, indent=4)


def task_training(params, name_model, blocks, selected_classes, dataset_sup, dataset_unsup, continual_learning, resume, task_num):
    
    dataset_sup["selected_classes"] = selected_classes
    dataset_unsup["selected_classes"] = selected_classes

    params.continual_learning = continual_learning
    params.resume = resume
    evaluate = False
    procedure(params, name_model, blocks, dataset_sup, dataset_unsup, evaluate, results, task_num)


def evaluation_phase(params, name_model, results, blocks, dataset_sup_ground, dataset_unsup_ground, cl_hyper):
# EVALUATION PHASE
    params.continual_learning = False
    params.resume = resume ################################################################################################################################
    evaluate = True
    if max(cl_hyper["evaluated_tasks"]) >= cl_hyper['n_tasks']:
        cl_hyper["evaluated_tasks"] = list(range(cl_hyper['n_tasks']))
    for task_num in cl_hyper["evaluated_tasks"]:
        print("################################## EVALUATION OF TASK " + str(task_num)+ " ############################################")

        selected_classes = cl_hyper["selected_classes"][task_num]

        dataset_sup_x = dataset_sup_ground.copy()
        dataset_unsup_x = dataset_unsup_ground.copy()

        dataset_sup_x["selected_classes"] = selected_classes
        dataset_unsup_x["selected_classes"] = selected_classes

        dataset_unsup_x["fold"] = fold + 1
        dataset_sup_x["fold"] = fold + 1
        
        dataset_sup_x["n_classes"] = len(selected_classes)
        dataset_unsup_x["n_classes"] = len(selected_classes)

        dataset_sup_x["out_channels"] = len(selected_classes)
        dataset_unsup_x["out_channels"] = len(selected_classes)

        procedure(params, name_model, blocks, dataset_sup_x, dataset_unsup_x, evaluate, results, task_num=task_num)

if __name__ == '__main__':


    
    params = parser.parse_args()
    folder_id = params.folder_id
    name_model = params.preset if params.model_name is None else params.model_name
    name_model = name_model + str(uuid.uuid4())
    #name_model = "C100_2C_CLb50abfcf-7c09-4b6f-a581-dc7b529dd310"
    blocks = load_presets(name=params.preset, preset_path=params.presets_path)
    classes_per_task = params.classes_per_task
    resume = params.resume
    
    
    cl_hyper = {
            'training_mode': params.training_mode,
            'cf_sol': params.cf_sol,
            'head_sol': params.head_sol,
            'top_k': params.top_k,
            "topk_lock": params.topk_lock,
            'high_lr': params.high_lr,
            'low_lr':params.low_lr,
            't_criteria': params.t_criteria,
            'delta_w_interval': params.delta_w_interval,
            "classes_per_task": params.classes_per_task,
            "n_tasks": params.n_tasks, 
            'selected_classes': eval(params.selected_classes),
            "evaluated_tasks": eval(params.evaluated_tasks), 
            "SINGLE": False

        }
        
    params.training_mode = cl_hyper["training_mode"]
    params.cl_hyper = cl_hyper
    
    
    dataset_sup_ground  = load_config_dataset(name=params.dataset_sup, validation=params.validation, cl=params.continual_learning, preset_path=params.presets_path)
    dataset_unsup_ground = load_config_dataset(name=params.dataset_unsup, validation=params.validation, cl=params.continual_learning, preset_path=params.presets_path)
    
    
    out_channels = dataset_sup_ground["out_channels"]
    dataset_sup_ground["old_dataset_size"] = dataset_sup_ground["width"]
    dataset_unsup_ground["old_dataset_size"] = dataset_unsup_ground["width"]

    dataset_sup_ground["n_classes"] = classes_per_task
    dataset_unsup_ground["n_classes"] = classes_per_task

    dataset_sup_ground["out_channels"] = classes_per_task
    dataset_unsup_ground["out_channels"] = classes_per_task

    dataset_sup_ground["joint"] = False
    dataset_unsup_ground["joint"] = False

    dataset_sup_ground["eval"] = False
    dataset_unsup_ground["eval"] = False
    
    if "ESC50" in params.dataset_sup:
        folds = 5
    elif "URBANSOUND8K" in params.dataset_sup:
        folds = 10
    # folds = 1 ##############################################################################################################################

    dataset_sup_1 = dataset_sup_ground.copy()
    dataset_unsup_1 = dataset_unsup_ground.copy()

    dataset_sup_1["joint"] = False
    dataset_unsup_1["joint"] = False

    results["performance_avg_folds"] = {}
    results["accuracy_matrix"] = {}
    results["joint"] = {}
    results["confusion_matrix"] = {}
    if out_channels >=  classes_per_task:
        for fold in range(folds):
            
            print("#########################################################################################################")
            print("################################## FOLD # " + str(fold+1)+ " ############################################")
            print("#########################################################################################################")
            dataset_unsup_1["fold"] = fold + 1
            dataset_sup_1["fold"] = fold + 1
            
            results["FOLD_#"+str(dataset_sup_1["fold"])] = {}
            # # TASK 1
            
            print("################################## TASK 0 ############################################")
            selected_classes = cl_hyper["selected_classes"][0]
            print("Selected Classes for the Task: ", selected_classes)
            dataset_sup_1["n_classes"] = len(selected_classes)
            dataset_unsup_1["n_classes"] = len(selected_classes)
            dataset_sup_1["out_channels"] = len(selected_classes)
            dataset_unsup_1["out_channels"] = len(selected_classes)


            task_training(params, name_model, blocks, selected_classes, dataset_sup_1, dataset_unsup_1, continual_learning=False, resume=None, task_num=0)
            evaluation_phase(params, name_model, results, blocks, dataset_sup_ground, dataset_unsup_ground, cl_hyper)
            

            
            for task_num in range(1, len(cl_hyper["selected_classes"])):
                print("################################## TASK " + str(task_num)+ " ############################################")

                selected_classes = cl_hyper["selected_classes"][task_num]
                print("Selected Classes for the Task: ", selected_classes)
                dataset_sup_x = dataset_sup_ground.copy()
                dataset_unsup_x = dataset_unsup_ground.copy()

                dataset_sup_x["joint"] = False
                dataset_unsup_x["joint"] = False

                dataset_sup_x["eval"] = False
                dataset_unsup_x["eval"] = False

                dataset_unsup_x["fold"] = fold + 1
                dataset_sup_x["fold"] = fold + 1
                dataset_sup_x["n_classes"] = len(selected_classes)
                dataset_unsup_x["n_classes"] = len(selected_classes)
                dataset_sup_x["out_channels"] = len(selected_classes)
                dataset_unsup_x["out_channels"] = len(selected_classes)

                params.continual_learning = True
                params.resume = True

                task_training(params, name_model, blocks, selected_classes, dataset_sup_x, dataset_unsup_x, continual_learning=True, resume=resume, task_num=task_num)
                # This evaluates all the tasks after training on a new task, needed to create the accuracy matrix.
                evaluation_phase(params, name_model, results, blocks, dataset_sup_ground, dataset_unsup_ground, cl_hyper)
                
                # # JOINT
                print(" ############################### JOINT PHASE at TASK " + str(task_num)+ " ############################")

                selected_classes = list(itertools.chain(*(cl_hyper["selected_classes"][:task_num+1])))     ### collapse all the lists into 1
                dataset_sup_x["joint"] = True
                dataset_unsup_x["joint"] = True
                params.resume = None
                dataset_unsup_x["fold"] = fold + 1
                dataset_sup_x["fold"] = fold + 1
                dataset_sup_x["n_classes"] = len(selected_classes)
                dataset_unsup_x["n_classes"] = len(selected_classes)
                dataset_sup_x["out_channels"] = len(selected_classes)
                dataset_unsup_x["out_channels"] = len(selected_classes)
                task_training(params, name_model+"joint", blocks, selected_classes, dataset_sup_x, dataset_unsup_x, continual_learning=True, resume=None, task_num=task_num)
                dataset_sup_x["eval"] = True
                params.resume = True
                procedure(params, name_model+"joint", blocks, dataset_sup_x, dataset_unsup_x, evaluate=True, results=results, task_num=task_num)
                dataset_sup_x["eval"] = False
                ###############################
            # EVALUATION PHASE

            evaluation_phase(params, name_model, results, blocks, dataset_sup_ground, dataset_unsup_ground, cl_hyper)


            results["count"] = 0

            # clean up the used models
            command = f"rm -rf -d {BASE_PATH}/Training/results/hebb/result/network/{name_model}"
            res = subprocess.run(command, shell=True, capture_output=False, text=True)
            command = f"rm -rf -d {BASE_PATH}/Training/results/hebb/result/network/{name_model+"joint"}"
            res = subprocess.run(command, shell=True, capture_output=False, text=True)

        results["model_name"] = name_model

        for task_num in range(0, cl_hyper["n_tasks"]):
            for fold_num in range(folds):
                
                if f"eval_{task_num}" not in results["performance_avg_folds"].keys():
                        results["performance_avg_folds"][f"eval_{task_num}"] = 0
                results["performance_avg_folds"][f"eval_{task_num}"] += results[f"FOLD_#{fold_num+1}"][f"eval_{task_num}"]["test_acc"]
            results["performance_avg_folds"][f"eval_{task_num}"] = results["performance_avg_folds"][f"eval_{task_num}"]/folds

        save_results(results, f"{params.parent_f_id}/TASKS_CL_{params.dataset_sup.split('_')[0] +  folder_id}", name_model)
        

    else: 
        print("Error: Not enough available classes to be organized in tasks of classes_per_task")