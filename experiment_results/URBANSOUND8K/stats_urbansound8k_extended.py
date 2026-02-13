import itertools
import os
import json

import numpy as np

def gather_data(path, report):
    
    for root, dirs, files in os.walk(path):
        for file in files:
            print(file)
            if ".json" not in file:
                continue
            with open(os.path.join(root, file), "r") as f:
                experiment = json.load(f)
                if experiment["cl_hyper"]["cf_sol"] == True:
                    key = "k_on"
                else: 
                    key = "k_off"
                
                for eval in experiment["performance_avg_folds"].keys():
                    if eval not in report[key]:
                        report[key][eval] = []
                    report[key][eval].append(experiment["performance_avg_folds"][eval])
                
                if "accuracy_matrix" not in report[key]:
                    report[key]["accuracy_matrix"] = {}

                for task_num in experiment["accuracy_matrix"].keys():

                    if f"T{task_num}" not in report[key]["accuracy_matrix"]:
                        report[key]["accuracy_matrix"][f"T{task_num}"] = []

                    for fold_num in experiment["accuracy_matrix"][task_num].keys():
                        report[key]["accuracy_matrix"][f"T{task_num}"].append(experiment["accuracy_matrix"][task_num][fold_num])

                # print(report[key]["accuracy_matrix"])
                if "joint" not in report[key]:
                    report[key]["joint"] = []

                for fold_num in experiment["joint"].keys():
                    report[key]["joint"].append(experiment["joint"][fold_num])
                
                if "im_steps" not in report[key]:
                    report[key]["im_steps"] = {}
                

                # "joint_confusion_matrix": {
                #     "FOLD_#1": {
                #         "T1": {
                #             "y_pred": [
                y_pred = []
                y_true = []
                for fold_num in experiment["confusion_matrix"]:
                    for task in experiment["confusion_matrix"][fold_num]:
                        for y in experiment["confusion_matrix"][fold_num][task]:
                            if y == "y_pred":
                                y_pred += experiment["confusion_matrix"][fold_num][task][y]
                            else:
                                y_true += experiment["confusion_matrix"][fold_num][task][y]

                for i in range(1, len(experiment["cl_hyper"]["selected_classes"])):
                    y_pred_tmp = np.array(y_pred)
                    y_true_tmp = np.array(y_true)
                    classes = list(itertools.chain(*experiment["cl_hyper"]["selected_classes"][:i+1]))
                    print(classes)
                    mask = np.isin(y_true_tmp, classes)

                    y_true_tmp = y_true_tmp[mask]
                    y_pred_tmp = y_pred_tmp[mask]

                    matches = np.sum(y_true_tmp == y_pred_tmp)
                    if i not in report[key]["im_steps"]:
                        report[key]["im_steps"][i] = []

                    report[key]["im_steps"][i].append(100*matches/len(y_true_tmp))


                

    for k in report.keys():
        # print(report[k]["joint"])
        if "joint" in report[k]:
            report[k]["joint"] = np.mean(np.array(report[k]["joint"]), axis=0).tolist()
        for i in range(1, 2):
            if "im_steps" not in report[k]:
                break
            report[k]["im_steps"][i] = np.mean(np.array(report[k]["im_steps"][i])).tolist()
        
        print("###########################")

    return report

def compute_BWT_aux(R, task_num):

    if task_num == 0:
        return None
    final_sum = 0.0
    for j in range(0, task_num):

        final_sum += R[task_num, j] - R[j, j]
    return final_sum/task_num

def compute_BWT(R):

    tasks = len(R)
    bwt = []
    for task_num in range(tasks):
        bwt.append(compute_BWT_aux(R, task_num))
    
    return bwt



def compute_IM(R, joint):
    tasks = len(R)
    final = []
    joint = [None]+joint

   

    for k in range(tasks):
        if joint[k] is None:
            final.append(None)
        else:
            final.append((joint[k]-R[k, k]).tolist())
    return final

def compute_FWT(R, b):
    return []

def compute_FM_aux(R, k):
    if k == 0:
        return None
    
    final_sum = 0.0
    for j in range(k):
        tmp = []
        for i in range(k):
            tmp.append(R[i,j]-R[k, j])
    final_sum += max(tmp)
    return final_sum/k

def compute_FM(R):
    tasks = len(R)
    final = []
    for k in range(tasks):
        final.append(compute_FM_aux(R, k))
    return final

def calculate_average_AM(report):
    accuracy_matrix = {"k_off": [], "k_on": []}
    for exp_type in report.keys():
        # print(exp_type)
        if report[exp_type] == {}:
            break
        for task_num in report[exp_type]["accuracy_matrix"]:
            
            # average across all folds
            t_avg_folds = np.array(report[exp_type]["accuracy_matrix"][task_num])
            t_avg_folds = np.mean(t_avg_folds, axis=0)
            # print(t_avg_folds)
            # t_col = np.array(report[exp_type]["accuracy_matrix"][task_num])
            # print(t_col)
            # t_col = np.mean(t_col, axis=0)
            accuracy_matrix[exp_type].append(t_avg_folds)
        accuracy_matrix[exp_type] = np.array(accuracy_matrix[exp_type])
        accuracy_matrix[exp_type] = accuracy_matrix[exp_type].T
    return accuracy_matrix

def calculate_average_evals(report):
    avg_accs = {"k_off":{}, "k_on": {}}
    for exp_type in report.keys():
        for key in report[exp_type].keys():
            if "eval" in key:
                task = key
                task_sum = 0
                counter = 0
                for acc in report[exp_type][task]:
                    task_sum += acc
                    counter += 1
                if task not in avg_accs[exp_type]:
                    avg_accs[exp_type][task] = task_sum/counter
    
    return avg_accs

def calculate_composite_avg(acc_matrix):
    """ Acc(up to task k, averaged across all previous tasks) [Acc (up to task k-1, 
    averaged across all previous tasks), Acc (on task k) ] """
    composite_accs = {"k_off":{}, "k_on": {}}
    for exp_type in composite_accs.keys():
        mat = acc_matrix[exp_type]
        for i in range(len(mat)):
            acc_curr = 0
            acc_upTok = 0
            acc_upTok_1 = 0
            for j in range(len(mat)):
                if i == j:
                    acc_curr = mat[i][j]
                    acc_upTok += mat[i][j]
                    break

                elif j < i:
                    acc_upTok += mat[i][j]
                    acc_upTok_1 += mat[i][j]
            if i == 0:
                composite_accs[exp_type][f"T{i}"]=[acc_upTok/(i+1), "-", acc_curr]
            else:
                composite_accs[exp_type][f"T{i}"]=[acc_upTok/(i+1),acc_upTok_1/(i), acc_curr]
    return composite_accs

def bootstrap_confidence_interval_aux(data, num_bootstrap=10000, confidence=0.95):
    data = np.array(data)
    means = []
    n = len(data)
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (confidence + (1 - confidence) / 2) * 100)
    return lower, upper

def bootstrap_confidence_interval(report, num_bootstrap=10000, confidence=0.95):
    """
    Compute the bootstrap confidence interval for the mean of the data.
    
    Parameters:
      data (array-like): The data vector (e.g., accuracies from one condition).
      num_bootstrap (int): Number of bootstrap samples (default 10,000).
      confidence (float): Confidence level (default 0.95).
      
    Returns:
      (lower, upper): Tuple with lower and upper bounds of the confidence interval.
    """
    confidence_intervals = {"k_off":{}, "k_on": {}}
    for exp_type in report.keys():
        for key in report[exp_type].keys():
            if "eval" in key:
                task = key
                lower, upper = bootstrap_confidence_interval_aux(report[exp_type][task])
                if task not in confidence_intervals[exp_type]:
                    confidence_intervals[exp_type][task] = 0
                confidence_intervals[exp_type][task] = (lower, upper)
    
    return confidence_intervals

path = input("Insert the path for the report: ")
report ={"k_off":{}, "k_on": {}}

report = gather_data(path, report)
avg_accs = calculate_average_evals(report)
confidence_intervals = bootstrap_confidence_interval(report)
R = calculate_average_AM(report)
composite_accs = calculate_composite_avg(R)
fm ={"k_off":{}, "k_on": {}}
fwt ={"k_off":{}, "k_on": {}}
im ={"k_off":{}, "k_on": {}}
bwt ={"k_off":{}, "k_on": {}}
for key in R.keys():
    if len(R[key]) == 0 or len(report[key])==0 :
        break
    fm[key] = compute_FM(R[key])
    FWT = []
    bwt[key] = compute_BWT(R[key])
    im[key] = compute_IM(R[key], list(report[key]["im_steps"].values()))
    R[key] = R[key].tolist()

# print(R)
# print(fm)
# print(bwt)
# print("report: ", report)
# print("avg_accs: ", avg_accs)
# print("confidence_intervals: ", confidence_intervals)

with open(os.path.join(path, "stats.txt"), "w") as f:
    f.write("avg_accs: "+ json.dumps(avg_accs, indent=4))
    f.write("composite_accs: "+ json.dumps(composite_accs, indent=4))
    f.write("confidence_intervals: "+ json.dumps(confidence_intervals, indent=4))
    f.write("bwt: "+ json.dumps(bwt, indent=4))
    f.write("im: "+ json.dumps(im, indent=4))
    f.write("fwt: "+ json.dumps({}, indent=4))
    f.write("fm: "+ json.dumps(fm, indent=4))
    f.write("Accuracy Matrix: "+ json.dumps(R, indent=4))
    f.write("report: "+json.dumps(report, indent=4))

