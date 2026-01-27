import itertools
import os
import json

import numpy as np

def gather_data(path, report):
    
    for root, dirs, files in os.walk(path):
        for file in files:
            print(file)
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
                

    return report


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
# confidence_intervals = bootstrap_confidence_interval(report)

with open(os.path.join(path, "stats.txt"), "w") as f:
    f.write("avg_accs: "+ json.dumps(avg_accs, indent=4))
    # f.write("confidence_intervals: "+ json.dumps(confidence_intervals, indent=4))
