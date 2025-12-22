import os
import json
import numpy as np



# Function to compute average accuracy and standard deviation
def average_behavior(path):
   
    statistics = {"KPM": {}, "M": {}}
    exp_num = 0
    name = None

    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if ".json" not in file:
                continue
            with open(os.path.join(root, file), "r") as f:
                json_obj = json.load(f)
                if json_obj["cl_hyper"]["cf_sol"] == True and json_obj["cl_hyper"]["head_sol"] == True:
                    cl_key = "KPM"
                    exp_num += 1
                elif json_obj["cl_hyper"]["cf_sol"] == False and json_obj["cl_hyper"]["head_sol"] == True:
                    cl_key = "M"
                else:
                    print(json_obj["cl_hyper"]["head_sol"], json_obj["cl_hyper"]["cf_sol"])
                    
                    continue
                
                for eval in json_obj["performance_avg_folds"].keys():
                    if eval not in statistics[cl_key]:
                        statistics[cl_key][eval] = 0
                    statistics[cl_key][eval] += json_obj["performance_avg_folds"][eval]
                

    for model_key in statistics.keys():
        for eval_key in statistics[model_key].keys():
            statistics[model_key][eval_key] = statistics[model_key][eval_key]/exp_num
    print(exp_num, statistics)
    # print(json.dumps(statistics))
    return statistics




dataset = "ESC50"
path = "/scratch/project_462001198/casciott/experiments/EXP_URBANSOUND8K_10C/TASKS_CL_URBANSOUND8K__second_run_5tasks"
id = path.split("/")[-1]
statistics = average_behavior(path)
with open(f"{path}/report.json", "w") as f:
    json.dump(statistics, f, indent=4)

# n_folds = len(statistics[list(statistics.keys())[0]].keys())
# acc_per_fold = accuracy_per_fold(statistics)
# acc_per_run = accuracy_per_run(statistics)
# graph_per_run(acc_per_run, id, n_folds)
# graph_per_fold(acc_per_fold, id, n_folds)
# sols = [(False, False), (True, False), (False, True),(True, True)]
# stats = []
# counter = 0
# important_stats = []
# eval_stats = []
# run_stats = []


# for sol in sols:
   
#     cl_hyper['cf_sol'] = sol[0]
#     cl_hyper['head_sol'] = sol[1]
    
    

#     if dataset == "C100": 
#         dataset = "CIFAR100"
#     elif dataset == "C10": 
#         dataset = "CIFAR10"
#     elif dataset == "IMG": 
#         dataset = "ImageNette"
#     elif dataset == "STL10": 
#         dataset = "STL10"
#     if dataset2 == "C100": 
#         dataset2 = "CIFAR100"
#     elif dataset2 == "C10": 
#         dataset2 = "CIFAR10"
#     elif dataset2 == "IMG": 
#         dataset2 = "ImageNette"
#     elif dataset2 == "STL10": 
#         dataset2 = "STL10"

#     if data_num == 1:
#         res = average_behavior(dataset, n_experiments, classes_per_task, n_tasks, f"{BASE_PATH}/SoftHebb-main/{parent_f_id}/TASKS_CL_{dataset+ folder_id}")
#     else:
#         res = average_behavior(dataset, n_experiments, classes_per_task, n_tasks, f"{BASE_PATH}/SoftHebb-main/{parent_f_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}")

#     #print(res)
#     stats.append(res)
     
    
    
# print("important_stats: ", important_stats)
# wilcoxon_test = []
# paired_test = []
# bootstrap_CI = []
# bootstrap_difference_CI = []
# cohen = []
# for key in stats[3]["eval_raw_stats"].keys(): 
#     print(len(stats[0]["eval_raw_stats"][key]), len(stats[3]["eval_raw_stats"][key]))
#     wilcoxon_test.append(wilcoxon_signed_rank_test(stats[3]["eval_raw_stats"][key], stats[2]["eval_raw_stats"][key]))
#     paired_test.append(paired_t_test(stats[3]["eval_raw_stats"][key], stats[2]["eval_raw_stats"][key]))
#     bootstrap_CI.append(bootstrap_confidence_interval(stats[3]["eval_raw_stats"][key]))
#     bootstrap_difference_CI.append(bootstrap_difference_ci(stats[3]["eval_raw_stats"][key], stats[2]["eval_raw_stats"][key]))
#     cohen.append(cohen_d_paired(stats[3]["eval_raw_stats"][key], stats[2]["eval_raw_stats"][key]))

# print("\n\n ######################################################################################\n\n")
# print("wilcoxon_test: ", wilcoxon_test)

# print("Paired t-test: ", paired_test)
    
#     # Bootstrap confidence interval for the mean of the continual learning accuracies
# print("Bootstrap CI for continual learning mean: ",bootstrap_CI )
    
#     # Bootstrap confidence interval for the difference in means
# print("Bootstrap CI for difference in means: ",  bootstrap_difference_CI)
    
#     # Cohen's d effect size
# print("Cohen's d effect size: ", cohen)
# print("\n\n ######################################################################################\n\n")

# # Create the plot
# plt.figure(figsize=(15, 7))
# plt.rcParams.update({'font.size': 20})
# plt.box(False)


# annotations = {}
# accuracies = {}
# # print("STATS: ", stats)

# for r in stats:
#     performances = r['performances']
#     avg_test_acc = r["avg_test_acc"]
#     std_test_acc = r["std_test_acc"]

#     r_keys = list(performances.keys())
#     r_keys = [r for r in r_keys if "" in r]
#     test_accs = [performances[r]['test_acc'] for r in r_keys]
#     std_accs = [performances[r]['std_test_acc'] for r in r_keys]

#     # Append avg_test_acc values
#     test_accs += [avg_test_acc[r] for r in avg_test_acc.keys()]
#     std_accs += [std_test_acc[r] for r in std_test_acc.keys()]
#     r_keys += list(avg_test_acc.keys())
#     print(r["avg_test_acc"])
#     print("LINE :", r_keys)
#     new_rkeys = []
#     for lab in r_keys: 
#         if "eval" in lab: 
#             new_rkeys.append(f"T{lab.split('_')[1]}")
#         else:
#             new_rkeys.append(lab)
#     r_keys = new_rkeys
#     #label = plt.text(0.50, 0.02, f"kernel solution={r['cl_hyper']['cf_sol']}, head solution={r['cl_hyper']['head_sol']}", horizontalalignment='left', wrap=True ,)
#     label = f"k={r['cl_hyper']['cf_sol']}, h={r['cl_hyper']['head_sol']}"
#     if label == "k=True, h=True":
#         label = "KPM-model"
#     elif label == "k=False, h=True":
#         label = "M-model"
#     elif label == "k=True, h=False":
#         label = "KP-model"
#     elif label == "k=False, h=False":
#         label = "V-model"

#     xs = list(range(len(r_keys)))
#     line, = plt.plot(xs, test_accs, marker='.', label=label, markersize=15)
#     accuracies[label] = []
#     for i in range(len(xs)):
#         accuracies[label].append((r_keys[i], test_accs[i] ))
    
#     color = line.get_color()
#     stat_annotations = {}
#     for x, (acc, std) in enumerate(zip(test_accs, std_accs)):
#         annotations.setdefault(x, []).append((acc, std, color))
#         stat_annotations.setdefault(x, []).append((acc, std, color, label))
#     temp = r
# print("ANNOTATIONS: ", annotations)
# print(xs)
# for i in range(len(r_keys)):
#     if "R" in r_keys[i]: 
#         r_keys[i] = "T"+r_keys[i][1:]
#     else:
#         r_keys[i] = "E"+r_keys[i][1:]
# plt.xticks(xs, r_keys, fontsize=30)
# for x, ann_list in annotations.items():
#     print(ann_list)
#     sorted_ann = ann_list
#     n = len(sorted_ann)
#     center = sum(a for a, s, c in sorted_ann) / n
#     spacing = 1
#     start_offset = -spacing * (n - 1) / 2
#     for i, (orig_acc, std, color) in enumerate(sorted_ann):
#         new_y = orig_acc + start_offset + i * spacing
#         print(x, len(annotations)//2, x%(len(annotations)//2), wilcoxon_test[x%(len(annotations)//2-1)][1])
#         if x > (len(annotations)//2 -1) and label == "KPM-model" and wilcoxon_test[x%(len(annotations)//2)][1] <= 0.055 and i == 3:
#             print("OK")
#             plt.text(x + 0.1, new_y, f'*', color=color, ha='left', va='center', fontsize=25)

# plt.xlabel('Training-Evaluation on Task #', fontsize=25)
# plt.ylabel('Test Accuracy', fontsize=25)
# plt.plot([0, len(annotations)-1], [100/classes_per_task, 100/classes_per_task], ':', lw=2, color="#ff0000", label="chance limit")

# #plt.title(f"{dataset} with {classes_per_task} classes per task, {n_tasks} tasks per experiment, on {res['count']} experiments")
# p_values = ""
# for i in range(len(wilcoxon_test)):
#     p_values += f"- evaluation on task {i}: {wilcoxon_test[i][1]}\n"
# #text = plt.text(0.50, 0.02, f'P-values between evaluations having both solutions on and just the head solution on:\n {p_values}', horizontalalignment='left', wrap=True ) 
# statistics = "{'wilcoxon_test': " + str(wilcoxon_test) +",\n" +"'Bootstrap':" + str(bootstrap_difference_CI)
# statistics += ",\n'Performances': " + json.dumps(accuracies, indent=4) + "}"
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# if data_num == 1: 
#     plt.savefig(f"{BASE_PATH}/SoftHebb-main/graphs/TASKS_CL_{id}_{dataset}_{n_tasks}T_{classes_per_task}C")
#     with open(f"{BASE_PATH}/SoftHebb-main/{parent_f_id}/TASKS_CL_{dataset+ folder_id}/TASKS_CL_{dataset+ folder_id}_statistics.txt", "w") as f: 
#         f.write(statistics)

# else:
#     plt.savefig(f"{BASE_PATH}/SoftHebb-main/{parent_f_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}", bbox_inches='tight')
#     with open(f"{BASE_PATH}/SoftHebb-main/{parent_f_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}/MULTD_CL_{dataset + '_' + dataset2  + '_' + folder_id}_statistics.txt", "w") as f: 
#         f.write(statistics)


# plt.close()
# #print("STATS: ", temp)
# create_boxplot_graph_eval(stats)
# create_boxplot_graph_runs(stats)