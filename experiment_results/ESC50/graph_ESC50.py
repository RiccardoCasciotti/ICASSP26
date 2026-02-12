import matplotlib.pyplot as plt
import numpy as np
csfont = {'fontname':'Times New Roman'}

# plt.title('title',**csfont)

# Data
avg_accs = {
    "Baseline": {
        "Task 0": 36.89166633605957,
        "Task 1": 70.75,
        "Task 2": 73.6,
        "Task 3": 79.2,
        "Task 4": 83.05
    },
    "KP-model": {
        "Task 0": 58.43333320617677,
        "Task 1": 82.05,
        "Task 2": 79.75,
        "Task 3": 80.05,
        "Task 4": 81.05
    }
}

tasks = list(avg_accs["Baseline"].keys())
x = np.arange(len(tasks))
width = 0.35

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9

# Plot
plt.figure(figsize=(8, 5))

# Use Times New Roman, font size 9pt
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9})

baseline_vals = list(avg_accs["Baseline"].values())
kp_vals = list(avg_accs["KP-model"].values())

plt.bar(x - width/2, baseline_vals, width, label="Baseline")
plt.bar(x + width/2, kp_vals, width, label="KP-model")

plt.xticks(x, tasks)
plt.ylabel("Accuracy (%)")
plt.xlabel("Tasks")
plt.legend()
plt.tight_layout()

plt.savefig("graph.png")
