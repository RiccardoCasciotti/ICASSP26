import os
import torch
import torch.nn as nn
import time
from sklearn.metrics import confusion_matrix
from PIL import ImageFile
import matplotlib.pyplot as plt
from mpltools import special
from copy import deepcopy
import pickle
import numpy as np
from utils import get_device
import os.path as op
try:
    from utils import RESULT, activation
except:
    from hebb.utils import RESULT, activation

ImageFile.LOAD_TRUNCATED_IMAGES = True

activations = {}
curr_layer = 0

POP_HEAD = True


def train_BP(model, criterion, optimizer, loader, device, measures):
    """
    Train only the traditional blocks with backprop
    """
    t = time.time()
    # model = model.to(device)
     
    DEVICE = get_device()

    
    for inputs, target in loader:
        ## 1. forward propagation$
        inputs = inputs.float().to(device, non_blocking=True)
        # 
        target = target.to(device, non_blocking=True)

        output = model(inputs)
        #  

        ## 2. loss calculation
        loss = criterion(output, target)
        
        ## 3. compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        ## 4. Accuracy assessment
        predict = output.data.max(1)[1]

        acc = predict.eq(target.data).sum()
        # Save if measurement is wanted
        

        convergence, R1 = model.convergence()
        measures.step(target.shape[0], loss.clone().detach().cpu(), acc.cpu(), convergence, R1, model.get_lr())

  
     
    return measures, optimizer.param_groups[0]['lr']


def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    
    activations[name] = torch.sum(output.detach().cpu(), dim=0)
    
    # ACTIVATIONS SHAPE:  torch.Size([10, 96, 32, 32]), where 10 is the batch size
    # what we have to do is sum all the activations to get 1 single kernel and we do this during all the training. At the end 
    # we create the semantic dictionary
     
  return hook


def get_layer(model, depth, prev_dict):
    # This function finds which layer is the training on, by basically checking if the state of then network is the same or not. 
    # When it finds one layer that has current state different than previous state then it returns the layer num and the prev_dict
    # cleaned of all the other non changing layers. 
    layer_num = 0
    total = len(model.config)
    for k in range(total):
        if prev_dict.get('blocks.' + str(k) + '.layer.weight') is not None:
            if not torch.equal(model.state_dict()['blocks.' + str(k) + '.layer.weight'], prev_dict['blocks.' + str(k) + '.layer.weight']):
                break
        layer_num += 1
    prev_dict = {k: v for k, v in prev_dict.items() if str(layer_num) in k}
     
    # 
    return prev_dict, layer_num

def get_delta_weights_bias(model, device, blocks, prev_dict ):
    #####
    # function which calculates the delta between the current state of the model and the previous state of the model. After doing so 
    # it stores the results in the delta_weights dictionary, which contains the delta weights of all the layers.
    #####
    i = 3
    curr_dict = deepcopy(model.state_dict())
    # 
    delta_bias = {}
    delta_weights = {}
    curr_dict = {k: v for k, v in curr_dict.items() if ".layer.weight" in k or ".layer.bias" in k and int(k.split(".")[1]) in blocks }
     

    # I should put all the tensors from the dict to a tensor which comprises all the layers
    # to improve performance by loading everything on GPU
    for kc, tc in curr_dict.items():
        tc = tc.to(device)
        tp = prev_dict[kc].to(device)
 
            # use subtract_() to do an inplace op and save space 
        tc.subtract_(tp)
        if i < 1:
             
             
            i +=1
            # !!!! double check if you need a deep copy or not
            # and also check if the tensor is in cpu or in gpu ...
        if  "bias" in kc:
            if kc not in delta_bias:
                delta_bias[kc] = []
            delta_bias[kc].append(tc.detach().clone())
        elif "weight" in kc: 
            if kc not in delta_weights:
                delta_weights[kc] = []
            delta_weights[kc].append(tc.detach().clone())
        del tc # removes the allocated gpu memory for tensor t
        del tp
        torch.cuda.empty_cache()# removes the reserved memory for tensor t
        
    return delta_weights, delta_bias

def get_delta_weights(model, device, blocks, depth, prev_dict, delta_weights ):
    #####
    # function which calculates the delta between the current state of the model and the previous state of the model. After doing so 
    # it stores the results in the delta_weights dictionary, which contains the delta weights of all the layers.
    #####
    i = 3
    curr_dict = deepcopy(model.state_dict())
    # 

    curr_dict = {k: v for k, v in curr_dict.items() if ".layer.weight" in k and int(k.split(".")[1]) in blocks }
     

    # I should put all the tensors from the dict to a tensor which comprises all the layers
    # to improve performance by loading everything on GPU
    for kc, tc in curr_dict.items():
        tc = tc.to(device)
        tp = prev_dict[kc].to(device)
                        
             
             
             
            # use subtract_() to do an inplace op and save space 
        tc.subtract_(tp)
        if i < 1:
             
             
            i +=1
            # !!!! double check if you need a deep copy or not
            # and also check if the tensor is in cpu or in gpu ...
        if  kc not in delta_weights:
            delta_weights[kc] = []
        delta_weights[kc].append(tc.detach().clone())
        del tc # removes the allocated gpu memory for tensor t
        del tp
        torch.cuda.empty_cache()# removes the reserved memory for tensor t
        
    return delta_weights

conv_act = []
images = []
def train_hebb(model, loader, device, blocks=[], measures=None, criterion=None):
    """
    Train only the hebbian blocks
    """
    
    t = time.time()
    
    loss_acc = (not model.is_hebbian()) and (criterion is not None)
    t = False
    i = 0

    file_path_d = 'avg_deltas.p'
    file_path_act = 'activations.p'
    
    avg_deltas = model.avg_deltas
    delta_weights = {}
    
    topk_kernels = model.topk_kernels

    t_criteria = model.cl_hyper["t_criteria"]
    layer_num = -1
    iteration = 0
    interval = model.cl_hyper["delta_w_interval"]
    depth = 0
    for layer in model.children():
        for subl in layer.children():
            depth += 1
    depth -= 1
    
    if t_criteria == "activations" and model.cl_hyper["cf_sol"]:
        prev_dict = deepcopy(model.state_dict())
        prev_dict = {k: v for k, v in prev_dict.items() if "layer.weight" in k and int(k.split(".")[1]) in blocks}
        activations_sum = {k: [] for k in prev_dict.keys() if int(k.split(".")[1]) in blocks}
        
     
     
         
    with torch.no_grad(): #Context-manager that disables gradient calculation.
        for inputs, target in loader:
            
           
            ## 1. forward propagation
            inputs = inputs.float().to(device)
            #  
            output = model(inputs) 
            
            if loss_acc:  
                target = target.to(device, non_blocking=True)
                 
                
                 
                ## 2. loss calculation
                loss = criterion(output, target)   

                ## 3. Accuracy assessment
                predict = output.data.max(1)[1]
                acc = predict.eq(target.data).sum()
                # Save if measurement is wanted
                conv, r1 = model.convergence()
                measures.step(target.shape[0], loss.clone().detach().cpu(), acc.cpu(), conv, r1, model.get_lr())
        
           
            model.update()
           
            # I store the activations of every batch
            if t_criteria == "activations" and model.cl_hyper["cf_sol"]:
                for k in list(prev_dict.keys()):
                    if int(k.split(".")[1]) in blocks:
                        #here we have to dive deeper on the sign of the weights... should we consider abs value once we summed all the cells in the kernel
                        # or at the beginning before doing the sum? Or maybe not consider abs values at all... ?

                        if len(activations_sum[k]) == 0: 
                            activations_sum[k].append(torch.abs(activations["conv" + k.split(".")[1]].cpu()))
                        else: 
                            activations_sum[k][0] += torch.abs(activations["conv" + k.split(".")[1]].cpu())




            if iteration % interval == 0 and model.cl_hyper["cf_sol"]: 

                delta_weights = get_delta_weights(model, device, blocks, depth, prev_dict, delta_weights)
                
                prev_dict = deepcopy(model.state_dict())
                prev_dict = {k: v for k, v in prev_dict.items() if ".layer.weight" in k and int(k.split(".")[1]) in blocks}     
           
                
            
            iteration += 1

    
    
   
    
    # here we sum all the values of each activation map to obtain 1 value of activation per kernel instead of a map.
    

    if t_criteria == "activations" and model.cl_hyper["cf_sol"]:
        for k in list(activations_sum.keys()):
                    if int(k.split(".")[1]) in blocks:
                        
                        activations_sum[k] = torch.sum(activations_sum[k][0], dim=1)
                        activations_sum[k] = torch.sum(activations_sum[k], dim=1)
                        
                        activations_sum[k] = {k:v for k, v in enumerate(activations_sum[k])}
        
                        activations_sum[k] = sorted(activations_sum[k].items(), key = lambda item : item[1], reverse=True)
                        activations_sum[k] = list(dict(activations_sum[k]))

        
        
                        K = round(len(activations_sum[k])*model.cl_hyper["top_k"]) # K takes #% of the kernels
                         
                        topk_kernels["conv" + k.split(".")[1]] = activations_sum[k][:K+1]
                         
                         
        model.topk_kernels = topk_kernels
     
        avg_deltas = average_deltas(delta_weights, avg_deltas, device)
    
        model.avg_deltas = avg_deltas                 
                         

    info = model.radius()
    convergence, R1 = model.convergence()
    with torch.no_grad():
        torch.cuda.empty_cache()

    return measures, model.get_lr(), info, convergence, R1


def average_deltas(delta_weights, avg_deltas,  device):
    # 
    summed_deltas = {}
    # 
    # 
    for k, v in delta_weights.items():
        res = torch.zeros(v[0].shape, device=device)
        for t in v:            
            res.add_(t)
            
        summed_deltas[k] = [len(v), res]
        # 

    # Now we sum all the cells 
    for  k, v in summed_deltas.items():
        channel_collapsed = torch.sum(v[1], 1)
        
        
        final_sum = torch.sum(channel_collapsed, (1,2))
        avg_tensor = final_sum / v[0]
        avg_deltas[k] = avg_tensor / max(avg_tensor) #normalize
        # 
    
    
    return avg_deltas


          

def train_sup_hebb(model, loader, device, measures=None, criterion=None, blocks=[]):
    """
    Train only the hebbian blocks

    """
    t = time.time()
    loss_acc = (not model.is_hebbian()) and (criterion is not None)
     
    t = False
    i = 0

    
    avg_deltas = model.avg_deltas
    delta_weights = {}
    activations_sum = []
    topk_kernels = model.topk_kernels

    
    layer_num = -1
    iteration = 0
    interval = model.cl_hyper["delta_w_interval"]
    depth = 0
    for layer in model.children():
        for subl in layer.children():
            depth += 1
    depth -= 1
        
    prev_dict = deepcopy(model.state_dict())
    prev_dict = {k: v for k, v in prev_dict.items() if "layer.weight" in k and int(k[7]) in blocks}
    activations_sum = {k: [] for k in prev_dict.keys() if int(k[7]) in blocks}
     
         


    with torch.no_grad():
        for inputs, target in loader:
            #  
            ## 1. forward propagation
            inputs = inputs.float().to(device)
            output = model(inputs)
            model.blocks[-1].layer.plasticity(x=model.blocks[-1].layer.forward_store['x'],
                                              pre_x=model.blocks[-1].layer.forward_store['pre_x'],
                                              wta=torch.nn.functional.one_hot(target, num_classes=
                                              model.blocks[-1].layer.forward_store['pre_x'].shape[1]).type(
                                              model.blocks[-1].layer.forward_store['pre_x'].type()))

            if loss_acc:

                target = target.to(device, non_blocking=True)

                ## 2. loss calculation
                loss = criterion(output, target)

                ## 3. Accuracy assessment
                predict = output.data.max(1)[1]
                acc = predict.eq(target.data).sum()
                # Save if measurement is wanted
                conv, r1 = model.convergence()
                measures.step(target.shape[0], loss.clone().detach().cpu(), acc.cpu(), conv, r1, model.get_lr())

            model.update()
            # if layer_num == -1:
            #     prev_dict, layer_num = get_layer(model, depth, prev_dict)
            
            # I store the activations of every batch
            l= 0
            for k in list(prev_dict.keys()):
                if int(k[7]) in blocks:
                    #here we have to dive deeper on the sign of the weights... should we consider abs value once we summed all the cells in the kernel
                    # or at the beginning before doing the sum? Or maybe not consider abs values at all... ?
                    if l == depth: 
                        if len(activations_sum[k]) == 0: 
                            activations_sum[k].append(activations["linear" + k[7]].cpu())
                        else:
                            activations_sum[k][0] += activations["linear" + k[7]].cpu()
                    else: 
                        if len(activations_sum[k]) == 0: 
                            activations_sum[k].append(activations["conv" + k[7]].cpu())
                        else: 
                            activations_sum[k][0] += activations["conv" + k[7]].cpu()
                    l += 1



            #remember that we are workng with batches, so you need to multiply interval by the batch size
            if iteration % interval == 0: 

                delta_weights = get_delta_weights(model, device, layer_num, depth, prev_dict, delta_weights)
                
                prev_dict = deepcopy(model.state_dict())
                #['blocks.0.operations.0.running_mean', 'blocks.0.operations.0.running_var', 'blocks.0.operations.0.num_batches_tracked', 'blocks.0.layer.weight', 'blocks.1.operations.0.running_mean', 'blocks.1.operations.0.running_var', 'blocks.1.operations.0.num_batches_tracked', 'blocks.1.layer.weight', 'blocks.2.operations.0.running_mean', 'blocks.2.operations.0.running_var', 'blocks.2.operations.0.num_batches_tracked', 'blocks.2.layer.weight', 'blocks.3.layer.weight', 'blocks.3.layer.bias']
                prev_dict = {k: v for k, v in prev_dict.items() if ".layer.weight" in k and int(k[7]) in blocks}
                
            
            iteration += 1

    # here we sum all the values of each activation map to obtain 1 value of activation per kernel instead of a map.
     


    for k in list(activations_sum.keys()):
                if int(k[7]) in blocks:
                    #here we have to dive deeper on the sign of the weights... should we consider abs value once we summed all the cells in the kernel
                    # or at the beginning before doing the sum? Or maybe not consider abs values at all... ?
                    activations_sum[k] = torch.sum(activations_sum[k][0], dim=1)
                    activations_sum[k] = torch.sum(activations_sum[k], dim=1)
                    # now we create a semantic dictionary associated with each activation, using the index of the kernel as key and the activation
                    # sum as value. Then we sort them, to consider only the first top k.
                    activations_sum[k] = {k:v for k, v in enumerate(activations_sum[k])}
    
                    activations_sum[k] = sorted(activations_sum[k].items(), key = lambda item : item[1], reverse=True)
                    activations_sum[k] = list(dict(activations_sum[k]))

    
    
                    K = round(len(activations_sum[k])*model.cl_hyper["top_k"]) # K takes 20% of the kernels
                     
                    topk_kernels["conv" + k[7]] = activations_sum[k][:K+1]

    avg_deltas = average_deltas(delta_weights, avg_deltas, device)
    
    model.avg_deltas = avg_deltas
    model.topk_kernels = topk_kernels


    info = model.radius()
    convergence, R1 = model.convergence()
    return measures, model.get_lr(), info, convergence, R1

""""""
def train_unsup(model, loader, device,
                blocks=[]):  # fixed bug as optimizer is not used or pass in the only use it has in this repo currently
    """
    Unsupervised learning only works with hebbian learning
    """
    model.train(blocks=blocks)  # set unsup blocks to train mode
    _, lr, info, convergence, R1 = train_hebb(model, loader, device, blocks=blocks)
    return lr, info, convergence, R1

"""
This function performs the training of the supervised learning part of the model.
The first thing we do is check if the number of blocks is = 1, but why??? 
Then we check if the first block is hebbian, if so we use train_sup_hebb().
otherwise it can be hybrid ( which implies tht there are more than just one block ) or simply the classical Back Prop.
"""
def train_sup(model, criterion, optimizer, loader, device, measures, learning_mode, blocks=[]):
    """
    train hybrid model.
    learning_mode=HB --> train_hebb
    learning_mode=BP --> train_BP
    """
    if len(blocks) == 1:
        model.train(blocks=blocks)
        # 
        if model.get_block(blocks[0]).is_hebbian():
            measures, lr, info, convergence, R1 = train_sup_hebb(model, loader, device, measures, criterion, blocks=blocks)
        else:
            measures, lr = train_BP(model, criterion, optimizer, loader, device, measures)
    else:
        model.train(blocks=blocks)
         
        if learning_mode == 'HB':
            measures, lr, info, convergence, R1 = train_sup_hebb(model, loader, device, measures, criterion, blocks=blocks)
        else:
            measures, lr = train_BP(model, criterion, optimizer, loader, device, measures)
    return measures, lr


def evaluate_unsup(model, train_loader, test_loader, device, blocks):
    """
    Unsupervised evaluation, only support MLP architecture

    """
     
    # 
    if model.get_block(blocks[-1]).arch == 'MLP':
        sub_model = model.sub_model(blocks)
        return evaluate_hebb(sub_model, train_loader, test_loader, device)
    else:
         

        return 0., 0.


def evaluate_hebb(model, train_loader, test_loader, device):
    if train_loader.dataset.split == 'unlabeled':
         
        return 0, 0
     

    preactivations, winner_ids, neuron_labels, targets = infer_dataset(model, train_loader, device)
    acc_train = get_accuracy(model, winner_ids, targets, preactivations, neuron_labels, device)

    preactivations_test, winner_ids_test, _, targets_test = infer_dataset(model, test_loader, device)
    acc_test = get_accuracy(model, winner_ids_test, targets_test, preactivations_test, neuron_labels, device)
    return float(acc_train.cpu()), float(acc_test.cpu())



def infer_dataset(model, loader, device):
    model.eval()
    targets_lst = []
    winner_ids = []
    preactivations_lst = []
     
    wta_lst = []
    with torch.no_grad():
        for inputs, targets in loader:
            ## 1. forward propagation
            inputs = inputs[targets != -1]
            targets = targets[targets != -1]
            if targets.nelement() != 0:
                inputs = inputs.float().to(device, non_blocking=True)
                preactivations, wta = model.foward_x_wta(inputs)
                
                preactivations_lst.append(preactivations)
                wta_lst.append(wta)
                targets_lst += targets.tolist()
                winner_ids_minibatch = wta.argmax(dim=1)
                winner_ids += winner_ids_minibatch.tolist()

    winner_ids = torch.FloatTensor(winner_ids).to(torch.int64).to(device)
    targets = torch.FloatTensor(targets_lst).to(torch.int64).to(device)
    preactivations = torch.cat(preactivations_lst).to(device)
    wta = torch.cat(wta_lst).to(device)
    neuron_labels = get_neuron_labels(model, winner_ids, targets, preactivations, wta)
    return preactivations, winner_ids, neuron_labels, targets


def evaluate_sup(model, criterion, loader, device, return_confusion_matrix=False):
    """
    Evaluate the model, returning loss and acc
    """
    model.eval()
    loss_sum = 0
    acc_sum = 0
    n_inputs = 0

    all_preds = []
    all_targets = []

     
    with torch.no_grad():
        for inputs, target in loader:
            ## 1. forward propagation
            inputs = inputs.float().to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
                 
            output = model(inputs)

            # print("target: ", torch.Tensor.tolist(target.cpu())[0])
            # print(torch.sort(output[0]))

            ## 2. loss calculation
            loss = criterion(output, target)
            loss_sum += loss.clone().detach()
            print("#######################################################")
            print("####################### LOSS: ", loss)
            print("####################### LOSS_SUM: ", loss)
            print("####################### n_inputs: ", n_inputs)
            print("#######################################################")

            ## 3. Accuracy assesment
            predict = output.data.max(1)[1]
            # print("predicted target: ", torch.Tensor.tolist(predict.cpu())[0])
            # print("#####################################################################")
            acc = predict.eq(target.data).sum()
            acc_sum += acc
            n_inputs += target.shape[0]
            print("target.shape[0]: ", target.shape[0])

            if return_confusion_matrix:
                all_preds.append(predict.cpu().detach().clone())
                all_targets.append(target.cpu().detach().clone())
             
    if return_confusion_matrix and not model.joint:
        y_pred = torch.cat(all_preds).numpy().tolist()
        y_true = torch.cat(all_targets).numpy().tolist()
        classes_offset = model.classes_offset

        if len(model.classes_offset) > 0:
            print("INSIDE OFFSET")
            for i in range(len(y_pred)):
                el_pred = y_pred[i]
                el_true = y_true[i]

                y_pred[i] = classes_offset[el_pred] + el_pred
                y_true[i] = classes_offset[el_true] + el_true

        obj = {"y_pred": y_pred, "y_true": y_true}
    
        # cm = confusion_matrix(y_true, y_pred)

        return loss_sum.cpu() / n_inputs, 100 * acc_sum.cpu() / n_inputs, obj
     

    return loss_sum.cpu() / n_inputs, 100 * acc_sum.cpu() / n_inputs

def evaluate_sup_multihead(model, criterion, loader, device, return_confusion_matrix=False):

    """
    Evaluate the multihead model, returning the best loss and acc
    """
    print("task_num: ", model.task_num)
    if POP_HEAD and not model.shmh: 
        # print("model.heads: ", model.heads)
        state_dict = model.state_dict()
        chosen_head = model.heads[model.task_num]
        

        # print("#################### CHOSEN HEAD ###############################")
        # print(len(model.heads), int(chosen_head[keys[1]].shape[0]), chosen_head, len(model.selected_classes), model.selected_classes)
        # print(int(chosen_head[keys[0]].shape[0]), len(model.selected_classes), model.selected_classes)
        # print("###################################################")

        if not op.isdir(RESULT):
            os.makedirs(RESULT)
        if not op.isdir(op.join(RESULT, 'network')):
            os.mkdir(op.join(RESULT, 'network'))
            os.mkdir(op.join(RESULT, 'layer'))

        folder_path = op.join(RESULT, 'network', model.model_name)
        if not op.isdir(folder_path):
            os.makedirs(op.join(folder_path, 'models'))
        storing_path = op.join(folder_path, 'models')
        torch.save({
        'state_dict': model.state_dict(),
        'config': model.config,
        'avg_deltas': model.avg_deltas,
        'topk_kernels': model.topk_kernels,
        'epoch': 50, 
        'heads': model.heads.copy(), 
        'heads_thresh' : model.heads_thresh, 
        'model_name': model.model_name
    }, op.join(storing_path, "checkpoint.pth.tar"))

        if chosen_head != None:
            keys = list(chosen_head.keys())
            state_dict[keys[0]] = chosen_head[keys[0]]
            state_dict[keys[1]] = chosen_head[keys[1]]

        # print("chosen_head: ", chosen_head)

        model.load_state_dict(state_dict)

        return evaluate_sup(model, criterion, loader, device, return_confusion_matrix=return_confusion_matrix)
    else:
        print("\n\n\nWARNING !!!! POP_HEAD AND SHMH ARE BOTH TRUE, THIS IS NOT SUPPORTED YET\n\n\n")



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_neuron_labels(model, winner_ids, targets, preactivations, wta):
    targets_onehot = nn.functional.one_hot(targets, num_classes=preactivations.shape[1]).to(torch.float32)
    winner_ids_onehot = nn.functional.one_hot(winner_ids, num_classes=preactivations.shape[1]).to(torch.float32)
    responses_matrix = torch.matmul(winner_ids_onehot.t(), targets_onehot)

    neuron_outputs_for_label_total = torch.matmul(wta.t(), targets_onehot)

    responses_matrix[responses_matrix.sum(dim=1) == 0] = neuron_outputs_for_label_total[
        responses_matrix.sum(dim=1) == 0]
    neuron_labels = responses_matrix.argmax(1)
    return neuron_labels


def get_accuracy(model, winner_ids, targets, preactivations, neuron_labels, device):
    n_samples = preactivations.shape[0]
    # if not model.ensemble:
    predlabels = torch.FloatTensor([neuron_labels[i] for i in winner_ids]).to(device)

    correct_pred = predlabels == targets
    n_correct = correct_pred.sum()
    accuracy = n_correct / len(targets)
    return 100 * accuracy.cpu()
