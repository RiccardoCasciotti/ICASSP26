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
    model.to(device)
   
    # if model.esc50:
    #     non_relevant_heads = list(set(range(50)) - set(model.selected_classes))

     
    DEVICE = get_device()

    
    for inputs, target in loader:
        ## 1. forward propagation$
        inputs = inputs.float().to(device, non_blocking=True)
        # 
        target = target.to(device, non_blocking=True)
        # if model.esc50:
        #     prev_dict = deepcopy(model.state_dict())
        #     prev_bias = {k: v for k, v in prev_dict.items() if "layer.bias" in k and int(k.split(".")[1]) in model.train_blocks}
        #     prev_weights = {k: v for k, v in prev_dict.items() if "layer.weight" in k and int(k.split(".")[1]) in model.train_blocks}

        #     prev_bias_els = np.array(prev_bias[list(prev_bias.keys())[-1]].clone().detach().cpu())
        #     prev_weights_els = np.array(prev_weights[list(prev_weights.keys())[-1]].clone().detach().cpu())



        output = model(inputs)
        #  

        ## 2. loss calculation
        loss = criterion(output, target)
        
        ## 3. compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if model.esc50:
            
        #     state_dict = deepcopy(model.state_dict())
            
            
        #     # print("prev_weights_els: ", prev_weights_els)
        #     linear_weights = np.array(state_dict[list(prev_weights.keys())[-1]].clone().detach().cpu())
            
        #     linear_weights[non_relevant_heads] = prev_weights_els[non_relevant_heads]
        #     # print("linear_weights: ", linear_weights)
        #     linear_bias = np.array(state_dict[list(prev_bias.keys())[-1]].clone().detach().cpu())
        #     linear_bias[non_relevant_heads] = prev_bias_els[non_relevant_heads]

        #     state_dict[list(prev_weights.keys())[-1]] = torch.tensor(linear_weights, device=DEVICE)
        #     state_dict[list(prev_bias.keys())[-1]] = torch.tensor(linear_bias, device=DEVICE)
            
        #     model.load_state_dict(state_dict)
            # print("prev_dict 0: ", prev_dict[list(prev_weights.keys())[-1]].clone().detach().cpu().tolist()[0])
            # print("curr_dict 0 : ", model.state_dict()[list(prev_weights.keys())[-1]].clone().detach().cpu().tolist()[0])

            # print("prev_dict 1: ", prev_dict[list(prev_weights.keys())[-1]].clone().detach().cpu().tolist()[1])
            # print("curr_dict 1 : ", model.state_dict()[list(prev_weights.keys())[-1]].clone().detach().cpu().tolist()[1])


        #          
        #          
        ###################################################################################

        # if layer_num == -1:
        #         prev_dict, layer_num = get_layer(model, depth, prev_dict)
            
        # # I store the activations of every batch
        # activations_sum.append(activations["linear" + str(layer_num)].cpu())

        # #remember that we are workng with batches, so you need to multiply interval by the batch size
        # if iteration % interval == 0: 

        #     delta_weights = get_delta_weights(model, device, layer_num, depth, prev_dict, delta_weights)
            
        #     
        #     #['blocks.0.operations.0.running_mean', 'blocks.0.operations.0.running_var', 'blocks.0.operations.0.num_batches_tracked', 'blocks.0.layer.weight', 'blocks.1.operations.0.running_mean', 'blocks.1.operations.0.running_var', 'blocks.1.operations.0.num_batches_tracked', 'blocks.1.layer.weight', 'blocks.2.operations.0.running_mean', 'blocks.2.operations.0.running_var', 'blocks.2.operations.0.num_batches_tracked', 'blocks.2.layer.weight', 'blocks.3.layer.weight', 'blocks.3.layer.bias']
        #     prev_dict = {k: v for k, v in prev_dict.items() if str(layer_num) + ".layer.weight" in k and str(depth) in k}     

                
            
        # iteration += 1
        ###################################################################################
        # Calculate the average weight change per neuron
        # Average change per row (neuron)
        # curr_weights = model.blocks[-1].layer.weight.detach().clone()
        # delta_weight = torch.abs(curr_weights - prev_weights)
        # avg_weight_change_per_neuron = torch.mean(delta_weight, dim=1)

        # # Forward pass again to compute activations
        # activations = output.detach().clone()  # Detach activations for analysis
        # # 
        # # Compute the importance of neurons based on average activation values
        # avg_activation_per_neuron = torch.mean(activations, dim=0) 

        ## 4. Accuracy assessment
        predict = output.data.max(1)[1]

        acc = predict.eq(target.data).sum()
        # Save if measurement is wanted
        

        convergence, R1 = model.convergence()
        measures.step(target.shape[0], loss.clone().detach().cpu(), acc.cpu(), convergence, R1, model.get_lr())

    
    #here we have to dive deeper on the sign of the weights... should we consider abs value once we summed all the cells in the kernel
    # or at the beginning before doing the sum? Or maybe not consider abs values at all... ?
    # final_sum = activations_sum[0]
    # for i in range(1, len(activations_sum)):
    #    final_sum += activations_sum[i]
    
    # # here we sum all the values of each activation map to obtain 1 value of activation per kernel instead of a map.ù
    #  
    # final_sum = torch.sum(final_sum, dim=0)
    # #final_sum = torch.sum(final_sum, dim=1)

    # # now we create a semantic dictionary associated with each activation, using the index of the kernel as key and the activation
    # # sum as value. Then we sort them, to consider only the first top k.
    # final_sum = {k:v for k, v in enumerate(final_sum)}
    
    # final_sum = sorted(final_sum.items(), key = lambda item : item[1], reverse=True)
    # final_sum = list(dict(final_sum))

    # K = round(len(final_sum)*0.3) # K takes 20% of the kernels
    #  
    # acts["conv" + str(layer_num)] = final_sum[:K+1]
    #  
    #  
    #  
    #  
    #  


    #  
    #  
    #  
    # avg_deltas = average_deltas(delta_weights, avg_deltas, device)
    #  
    #  

    
    # model.avg_deltas = avg_deltas
    # model.acts = acts

    #  
    #  

    #  
    t_criteria = model.cl_hyper["t_criteria"]
    topk_kernels = model.topk_kernels
    num_blocks = len(topk_kernels) + 1
     
    # if t_criteria == "KSE":
    #     weights = deepcopy(model.state_dict())
    #     weights = {int(k[7]): v for k, v in weights.items() if ".layer.weight" in k} 
    #     kse_indicators = compute_kse_indicator(weights)
    #      
    #      
    #     for layer in kse_indicators.keys():
    #          
    #          
    #         if layer== (num_blocks-1):
    #              
    #             kernels = {k:v for k, v in enumerate(kse_indicators[layer])}
    #             kernels = sorted(kernels.items(), key = lambda item : item[1], reverse=True)
    #             kernels = list(dict(kernels))Y
    #             K = round(len(kernels)*model.cl_hyper["top_k"]) # K takes 20% of the kernels
    #             topk_kernels["conv" + str(layer)] = kernels[:K+1]
    #              
    # model.topk_kernels = topk_kernels
    #  
    # for k,v in topk_kernels.items():

    #      

        
    return measures, optimizer.param_groups[0]['lr']

"""
The first thing we do is check if the model is hebbian or not (basically if it is we set the loss accuracy to False).
Then we tell torch not to calculate any gradient because we don't need any for unsupervised hebbian. 
We don't get inside the if loss_acc clause why???
model.is_hebbian() returns true if the last block of the model is hebbian or not and checks if the criterion is not none.
The criterion can be something like ... ??? criterion seems to be none always, just like measures. 
So are they both to be defined??? 


"""


def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    
    activations[name] = torch.sum(output.detach().clone(), dim=0)
    
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

def compute_kse_indicator(weights_tot, alpha=1.0, k=5, normalize=False, eps=1e-8):
    """
    Compute the Kernel Sparsity and Entropy (KSE) indicator for a convolutional layer.
    
    Args:
        weights (torch.Tensor): Convolutional kernel weights of shape (N, C, Kh, Kw),
                                where N is the number of filters (output channels),
                                C is the number of input channels, and Kh, Kw are the kernel height and width.
        alpha (float): Balance parameter to control the influence of entropy (default: 1.0).
        k (int): Number of nearest neighbors to use when computing kernel entropy (default: 5).
        normalize (bool): If True, apply min-max normalization to rescale the indicator to [0, 1].
        eps (float): A small value to avoid division by zero.
    
    Returns:
        torch.Tensor: A 1D tensor of length C containing the KSE indicator for each input channel.
    """
    kse_indicators_tot = {}
    for i in range(len(weights_tot)):

        weights =  weights_tot[i]
        
        # Get dimensions: N = number of kernels per input channel, C = number of input channels.
        N, C, Kh, Kw = weights.shape
        # Prepare tensor to store the indicator for each input channel.
        kse_indicators = torch.zeros(C, device=weights.device)
        
        # Loop over each input channel
        for c in range(C):
            # Get all 2D kernels corresponding to input channel c and flatten each kernel
            kernels = weights[:, c, :, :].view(N, -1)  # Shape: (N, Kh*Kw)
            
            # Compute kernel sparsity as the sum of L1 norms of the kernels.
            # Note: You could also compute each kernel's L1 norm and then sum them.
            s_c = torch.sum(torch.abs(kernels))
            
            # Compute pairwise Euclidean distances between kernels.
            # This results in an (N, N) distance matrix.
            dist_matrix = torch.cdist(kernels, kernels, p=2)
            
            # For each kernel, ignore self-distance by setting the diagonal to infinity.
            inf_diag = torch.full((N,), float('inf'), device=weights.device)
            dist_matrix.fill_diagonal_(float('inf'))
            
            # For each kernel (each row in the distance matrix), get the distances of its k nearest neighbors.
            knn_distances, _ = torch.topk(dist_matrix, k=k, largest=False)
            
            # Compute a density metric for each kernel as the sum of its k nearest neighbor distances.
            density = torch.sum(knn_distances, dim=1)  # Shape: (N,)
            
            # Total density for the input channel c
            d_c = torch.sum(density)
            
            # Compute kernel entropy for this channel.
            if d_c.item() == 0:
                entropy = torch.tensor(0.0, device=weights.device)
            else:
                p_i = density / (d_c + eps)
                # Avoid log2(0) by adding a small epsilon.
                entropy = -torch.sum(p_i * torch.log2(p_i + eps))
            
            # Combine sparsity and entropy into the KSE indicator:
            # v_c = sqrt( s_c / (1 + alpha * entropy) )

            #v_c = torch.sqrt(s_c / (1 + alpha * entropy + eps))
            #kse_indicators[c] = v_c
            kse_indicators[c] = entropy
        
        # Optionally, normalize the indicators to [0, 1]
        if normalize:
            min_val = torch.min(kse_indicators)
            max_val = torch.max(kse_indicators)
            kse_indicators = (kse_indicators - min_val) / (max_val - min_val + eps)
        if len(kse_indicators) > 3: 
            kse_indicators_tot[i-1] = kse_indicators
    return kse_indicators_tot

def train_hebb(model, loader, device, blocks=[], measures=None, criterion=None):
    """
    Train only the hebbian blocks
    """
    
    t = time.time()
    # 
    # 
    # 
    loss_acc = (not model.is_hebbian()) and (criterion is not None)
    t = False
    i = 0

    file_path_d = 'avg_deltas.p'
    file_path_act = 'activations.p'
    
    avg_deltas = model.avg_deltas
    delta_weights = {}
    
    topk_kernels = model.topk_kernels

    #Check if the file exists
    # if os.path.exists(file_path_d):
    #     with open('avg_deltas.p', 'rb') as pfile:
    #         avg_deltas = pickle.load(pfile)
    # else:
    #     avg_deltas = {}
    #     with open('avg_deltas.p', 'wb') as pfile:
    #         pickle.dump(avg_deltas, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    
    # #Check if the file exists
    # if os.path.exists(file_path_act):
    #     with open('activations.p', 'rb') as pfile:
    #         topk_kernels = pickle.load(pfile)
    # else:
    #     with open('activations.p', 'wb') as pfile:
    #         pickle.dump(topk_kernels, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    t_criteria = model.cl_hyper["t_criteria"]
    layer_num = -1
    iteration = 0
    interval = model.cl_hyper["delta_w_interval"]
    depth = 0
    for layer in model.children():
        for subl in layer.children():
            depth += 1
    depth -= 1
        
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
            if t_criteria == "activations":
                for k in list(prev_dict.keys()):
                    if int(k.split(".")[1]) in blocks:
                        #here we have to dive deeper on the sign of the weights... should we consider abs value once we summed all the cells in the kernel
                        # or at the beginning before doing the sum? Or maybe not consider abs values at all... ?

                        if len(activations_sum[k]) == 0: 
                            activations_sum[k].append(torch.abs(activations["conv" + k.split(".")[1]].cpu()))
                        else: 
                            activations_sum[k][0] += torch.abs(activations["conv" + k.split(".")[1]].cpu())




            #remember that we are workng with batches, so you need to multiply interval by the batch size
            if iteration % interval == 0: 

                delta_weights = get_delta_weights(model, device, blocks, depth, prev_dict, delta_weights)
                
                prev_dict = deepcopy(model.state_dict())
                #['blocks.0.operations.0.running_mean', 'blocks.0.operations.0.running_var', 'blocks.0.operations.0.num_batches_tracked', 'blocks.0.layer.weight', 'blocks.1.operations.0.running_mean', 'blocks.1.operations.0.running_var', 'blocks.1.operations.0.num_batches_tracked', 'blocks.1.layer.weight', 'blocks.2.operations.0.running_mean', 'blocks.2.operations.0.running_var', 'blocks.2.operations.0.num_batches_tracked', 'blocks.2.layer.weight', 'blocks.3.layer.weight', 'blocks.3.layer.bias']
                prev_dict = {k: v for k, v in prev_dict.items() if ".layer.weight" in k and int(k.split(".")[1]) in blocks}     
           
                
            
            iteration += 1

    
    
   
    
    # here we sum all the values of each activation map to obtain 1 value of activation per kernel instead of a map.
    # 

    if t_criteria == "activations":
        for k in list(activations_sum.keys()):
                    if int(k.split(".")[1]) in blocks:
                        #here we have to dive deeper on the sign of the weights... should we consider abs value once we summed all the cells in the kernel
                        # or at the beginning before doing the sum? Or maybe not consider abs values at all... ?
                        activations_sum[k] = torch.sum(activations_sum[k][0], dim=1)
                        activations_sum[k] = torch.sum(activations_sum[k], dim=1)
                        # now we create a semantic dictionary associated with each activation, using the index of the kernel as key and the activation
                        # sum as value. Then we sort them, to consider only the first top k.
                        activations_sum[k] = {k:v for k, v in enumerate(activations_sum[k])}
        
                        activations_sum[k] = sorted(activations_sum[k].items(), key = lambda item : item[1], reverse=True)
                        activations_sum[k] = list(dict(activations_sum[k]))

        
        
                        K = round(len(activations_sum[k])*model.cl_hyper["top_k"]) # K takes #% of the kernels
                         
                        topk_kernels["conv" + k.split(".")[1]] = activations_sum[k][:K+1]
                         
                         
                         
                         
                         
    elif t_criteria == "KSE":
        weights = deepcopy(model.state_dict())
        weights = {int(k.split(".")[1]): v for k, v in weights.items() if ".layer.weight" in k and int(k.split(".")[1]) in blocks} 
        kse_indicators = compute_kse_indicator(weights)
         
        for layer in kse_indicators.keys():
            if layer in blocks:
                kernels = {k:v for k, v in enumerate(kse_indicators[layer])}
                kernels = sorted(kernels.items(), key = lambda item : item[1], reverse=False)
                kernels = list(dict(kernels))
                K = round(len(kernels)*model.cl_hyper["top_k"]) # K takes 20% of the kernels
                topk_kernels["conv" + str(layer)] = kernels[:K+1]
                 
     
    # last = int((list(topk_kernels.keys())[-1])[-1]) + 1
    # topk_kernels["conv" + str(last)] = topk_kernels["conv" + str(last-1)] 
    # topk_kernels.pop(list(topk_kernels.keys())[0])
    model.topk_kernels = topk_kernels

     
     
     
    avg_deltas = average_deltas(delta_weights, avg_deltas, device)
     
     

    
    model.avg_deltas = avg_deltas
    

     


    
    info = model.radius()
    convergence, R1 = model.convergence()
    with torch.no_grad():
        torch.cuda.empty_cache()

    return measures, model.get_lr(), info, convergence, R1

""" 
Again we set the gradient to zero.
There is still the problem with the criterion which looks like to be always none and not defined anywhere.
We then load the input to the device and calculate the output.

The model.blocks.plasticity function allow to calculate the weight change vector which is done only the last layer. 
why??
Let's reason a little bit, we have to update the weights, now this could either be something which dpenedds on the type of
ùoperations that we are performing or it could be something which derives from the nature of the input itself. Could it be that 
since we are working with networks which have at most one hebbian layer??? I don't understand, this need to be investigated 
a bit more. 

Here we don't have the same issue we have for the unsupervised learning where we never get into the if loss_acc clause 
because the criterion and the measures are never passed. In this case the criterion passed is  criterion = nn.CrossEntropyLoss()
and the measures is   log_batch = log.new_log_batch() which is defined in run_sup() and needs to be investigated.
There is the possibility of entering the clause but I don't understand how we manage to do it: 
the model is supposed to be not hebbian to calculate the loss... right, but if all the models we 
consider are hebbian then what? this is train sup hebbian function... wth it doesn't make sense. 
Let's try and analyze it: 
to enter the if clause we need to not be hebbian, which is set through the flag is_hebbian, which is set to true
if  when we read the preset flag in the object contained in the preset.json we read anyhing but MLP. Then we need
to have a criterion which is not none and we aldready saw that when we call it the object passed is the 
cross entropy loss criteria. So we just need to understand when the flag for hebbian is set to false. Like what is 
the role of this function because from my initial understanding it was to train the hebbian learning in a 
supervised manner, which kindd of doesn't make sense because if we are working with an hebbian network where
should we utilize the feedback given from the supervised approach? This is used only in the case the model is not hebbian, 
So now the question becomes when is the model not hebbian??
The is_hebbian function returns true by checking only the last block. If the last block has the hebbian flag set to true
then is_hebbian return  true, then when is the last layer set to hebbian? We check if the preset is field is either soft or BP. 
If it is BP we set the hebbian flag to false otherwise we set it to true. 

"2SoftMlpMNIST": {
      "b0": {
        "arch": "MLP",
        "preset": "soft-c2000-t12-lr0.045-r35-v1",
        "operation": "flatten",
        "activation": "softmax_5",
        "num": 0,
        "batch_norm": false
      },
      "b1": {
        "arch": "MLP",
        "operation": "",
        "preset": "BP-c10",      ------------------------------------> we check this field here 
        "dropout": 0,
        "num": 1
      }

      Ok so we got how the whole thing works, but then why are we just using 1 block?? 
      By this I mean that if the check is done on the last block only and this block is hebbian because we dont get in the
      if then we must be using only one block not considerign the second one which is always using back prop for classification.

    UPDATE: when we call this train_sup_hebb the is_hebbian is set to false... almost looks like it splits the model down in
    two parts when we have to train the hebbian part we call the run_unsup and when train the classificator we call run_sup, 
    which is why the length of blocks is just one... the classificator block is always one! Ok so the division is done in ray search when we
    check in the config if the mode is unsupervised or supervised. 



"""

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

     
    # 

     


    
    

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


"""
we take the model and call eval to turn off batch normalization layers, dropout and so on 
because we need to put the model in inference mode. 
We then take all the labels ( targets ).
Then we load all the input to the gpu setting the non blocking flag to true: 
the non_blocking flag is used in data transfer operations between CPU and GPU memory. 
When this flag is set to True, it allows the transfer to be asynchronous, meaning it does not 
block the execution of the program while waiting for the data transfer to complete.

After that we take the preactivations and the wta from the forward_x_wta
wta: are th
"""
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
                # 
                # 
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

def head_choser(model, criterion, loader, device):
    """
    Evaluate the model on a small batch to pick the best head to mount
    """
    model.eval()
    loss_sum = 0
    acc_sum = 0
    n_inputs = 0
    tot_sum = 0
    with torch.no_grad():
        for inputs, target in loader:
            ## 1. forward propagation
            inputs = inputs.float().to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(inputs)

            tot_sum_tmp, indexes = torch.max(output.detach().clone() ,dim=1)

            top2_outputs, top_indexes = torch.topk(output.detach().clone(), 2, dim=1, largest=True, sorted=True)
            top2_outputs = top2_outputs[:,1]
            # print("top2_outputs: ", top2_outputs[:5])
            # print("tot_sum_tmp: ", tot_sum_tmp[:5])       
            # print("top2_outputs.shape: ", top2_outputs.shape)
            # print("tot_sum_tmp.shape: ", tot_sum_tmp.shape)
            tot_sum_tmp = torch.sub(tot_sum_tmp, top2_outputs)
            # print("tot_sum_tmp.shape 2: ", tot_sum_tmp.shape)
            # print("tot_sum_tmp 2: ", tot_sum_tmp[:5])       


            tot_sum += torch.sum(tot_sum_tmp, dim=0)

    return tot_sum

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
                 
                 
            ## 3. Accuracy assesment
            predict = output.data.max(1)[1]
            # print("predicted target: ", torch.Tensor.tolist(predict.cpu())[0])
            # print("#####################################################################")
            acc = predict.eq(target.data).sum()
            acc_sum += acc
            n_inputs += target.shape[0]

            if return_confusion_matrix:
                all_preds.append(predict.cpu())
                all_targets.append(target.cpu())
             
    if return_confusion_matrix:
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        cm = confusion_matrix(y_true, y_pred)
        return loss_sum.cpu() / n_inputs, 100 * acc_sum.cpu() / n_inputs, cm
     

    return loss_sum.cpu() / n_inputs, 100 * acc_sum.cpu() / n_inputs

def shmh_fuser(heads):
    weights = []
    biases = []
    head = {}
    for h in heads:
        keys = list(h.keys())
        if keys[0] not in head.keys():
            head[keys[0]] = []
        if keys[1] not in head.keys():
            head[keys[1]] = []
        head[keys[0]].append(h[keys[0]])
        head[keys[1]].append(h[keys[1]])
    keys = list(head.keys())
    head[keys[0]] = torch.cat(head[keys[0]], dim=0)
    head[keys[1]] = torch.cat(head[keys[1]], dim=0)

    return head

def evaluate_sup_multihead(model, criterion, loader, device):

    """
    Evaluate the multihead model, returning the best loss and acc
    """
    if POP_HEAD and not model.shmh: 
        # print("model.heads: ", model.heads)
        state_dict = model.state_dict()
        chosen_head = model.heads[0]
        keys = list(chosen_head.keys())
        model.heads = model.heads[1:]

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

        keys = list(chosen_head.keys())
        state_dict[keys[0]] = chosen_head[keys[0]]
        state_dict[keys[1]] = chosen_head[keys[1]]

        # print("chosen_head: ", chosen_head)

        model.load_state_dict(state_dict)

        return evaluate_sup(model, criterion, loader, device)
    else:
        print("\n\n\nWARNING !!!! POP_HEAD AND SHMH ARE BOTH TRUE, THIS IS NOT SUPPORTED YET\n\n\n")
    if model.shmh: 
        state_dict = model.state_dict()
        head = shmh_fuser(model.heads)
        keys = list(head.keys())
        state_dict[keys[0]] = head[keys[0]]
        state_dict[keys[1]] = head[keys[1]]
        model.load_state_dict(state_dict)
        return evaluate_sup(model, criterion, loader, device)
    

    heads_performance = {}
    head_num = 0

    for head in model.heads:
        # model.to(get_device())
        state_dict = model.state_dict()
        chosen_head = head
        keys = list(chosen_head.keys())
        state_dict[keys[0]] = chosen_head[keys[0]]
        state_dict[keys[1]] = chosen_head[keys[1]]
        
        # print("#################### CHOSEN HEAD ###############################")
        # print(int(chosen_head[keys[1]].shape[0]), chosen_head, len(model.selected_classes), model.selected_classes)
        # print(int(chosen_head[keys[0]].shape[0]), len(model.selected_classes), model.selected_classes)
        # print("###################################################")

        if int(chosen_head[keys[0]].shape[0]) != len(model.selected_classes):
            heads_performance[head_num] = 0
            head_num += 1
            continue
        model.load_state_dict(state_dict)
        model.eval()

        tot_sum = head_choser(model, criterion, loader, device)
        heads_performance[head_num] = tot_sum


        head_num += 1

    max_val = 0
    max_key = 0
    # print("heads: ", heads_performance)
    for k, i in heads_performance.items():
        if i > max_val:
            max_val = i
            max_key = k
            
     

    chosen_head = model.heads[max_key]
    # print("chosen_head: ", max_key)
    if POP_HEAD: 
        chosen_head = model.heads[0]
        model.heads = model.heads[1:]

    keys = list(chosen_head.keys())
    state_dict[keys[0]] = chosen_head[keys[0]]
    state_dict[keys[1]] = chosen_head[keys[1]]
    model.load_state_dict(state_dict)
   

    
    return evaluate_sup(model, criterion, loader, device)


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


"""Code needs to be rewrite"""


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
    '''
    else:
        if model.test_uses_softmax:
            soft_acts = activation(preactivations, model.t_invert, model.activation_fn, dim=1, power=model.power, normalize=True)
            winner_ensembles = [
                np.argmax([np.sum(np.where(neuron_labels == ensemble, soft_acts[sample], np.asarray(0))) for
                           ensemble in range(10)]) for sample in range(n_samples)]
        else:
            winner_ensembles = [
                np.argmax([np.sum(np.where(neuron_labels == ensemble, preactivations[sample], np.asarray(0))) for
                           ensemble in range(10)]) for sample in range(n_samples)]
        predlabels = winner_ensembles
    '''
    correct_pred = predlabels == targets
    n_correct = correct_pred.sum()
    accuracy = n_correct / len(targets)
    return 100 * accuracy.cpu()
