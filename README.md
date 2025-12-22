## 🧑‍🎓 Author

Riccardo Casciotti, Tampere University 
Co-Authors: Prof. Alberto Antonietti - Politecnico di Milano, Francesco De Santis - Politecnico di Milano, Prof. Annamaria Mesaros - Tampere University 


# INCREMENTAL LEARNING FOR AUDIO CLASSIFICATION WITH HEBBIAN DEEP NEURAL NETWORKS

The ability of humans for lifelong learning is an inspiration for
deep learning methods and in particular for continual learning. In
this work, we apply Hebbian learning, a biologically inspired learn-
ing process, to sound classification. We propose a kernel plasticity
approach that selectively modulates network kernels during incre-
mental learning, acting on selected kernels to learn new informa-
tion and on others to retain previous knowledge. Using the ESC-
50 dataset, the proposed method achieves 76.3% overall accuracy
over five incremental steps, outperforming a baseline without kernel
plasticity (68.7%) and demonstrating significantly greater stability
across tasks.
## 📄 Overview

Modern Artificial Neural Networks (ANNs) still struggle with **catastrophic forgetting** — the tendency to forget previously learned tasks when learning new ones. Inspired by **Hebbian learning** and **neuromodulation** mechanisms found in the human brain, this project implements novel mechanisms within a biologically plausible architecture, **SoftHebb**, to mitigate forgetting in a **task-free continual learning** setup.

## 🚀 Objectives

- Implement and test **neuromodulation-inspired plasticity** control mechanisms.
- Apply a **multi-head architecture** to isolate task-specific learning at the classifier level.
- Validate the model's performance on **incremental sound classification tasks** using ESC-50.

## 🧪 Methods

### 🧬 SoftHebb Architecture
- A deep convolutional neural network trained in an **unsupervised** manner based on correlations in the input.
- Mimics **Hebbian synaptic plasticity** rules.

### 🔁 Kernel Plasticity Neuromodulation
A dopamine-inspired approach for selective weight update:
1. **Track kernel weight changes** over training intervals.
2. **Rank kernels** by their cumulative activation values.
3. **Modulate learning** by:
   - Reducing plasticity of important kernels.
   - Increasing plasticity for less relevant ones.

### 🧠 Multi-Head Architecture
- Each task has its own **head** (final classifier layer).
- During inference, the model automatically selects the most appropriate head using an unsupervised scoring mechanism.

## 📈 Key Results

- The **KPM-model** outperforms all other variants in retaining earlier task performance.
- It **balances memory retention and adaptability**, addressing catastrophic forgetting effectively.
- Gains were most significant with 6-layer networks and moderate task complexity.
- Models converge in **one unsupervised epoch**, offering **fast training** and **efficient learning**.


# 📦 Setup Instructions
## 📁 Project Structure

```
neuromodAI-main/
├── SoftHebb-main/           # Core Hebbian model and engine code
│   ├── train.py             # Main training pipeline
│   └── model.py             # SoftHebb model definition
├── batches/                 # Experiment automation scripts and testing
│   ├── testing.py           # Evaluation scripts for experiments
│   ├── t_hyper.py           # Hyperparameter tuning or task-specific setup
│   ├── stats.py             # Statistical analysis of experimental results
│   └── latex_tables.py      # LaTeX table generation for paper-ready output
├── softhebb_env/            # Conda environment files
│   ├── conda_reqs.txt
│   └── pip_reqs.txt
```

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/neuromodAI.git
cd neuromodAI/neuromodAI-main
```

### 2. Create and Activate the Conda Environment

```bash
conda create --name softhebb_env python=3.8
conda activate softhebb_env
pip install -r softhebb_env/pip_reqs.txt
```


## 📊 Model Testing & Configuration (via `t_hyper.py`)

The script `t_hyper.py` in the `batches/` folder is used to configure and manage continual learning experiments, particularly for hyperparameter tuning or automated experiment runs across different datasets and task configurations.
### 📅 Batch Job Execution

All training experiments are designed to be executed via **batch job scripts**, provided in the `batches/` folder. These scripts are particularly suited for running on clusters with job scheduling systems (e.g., SLURM).

### 🛠️ Configuration Parameters

These are the main parameters used in `t_hyper.py`:

| Parameter            | Description |
|---------------------|-------------|
| `classes_per_task`  | Number of classes associated with each task (e.g., 2, 4, 6). |
| `n_experiments`     | Number of repeated runs per configuration. Default: 80. |
| `n_tasks`           | Total number of tasks to be learned incrementally. |
| `evaluated_tasks`   | List of task indices to evaluate performance on. |
| `data_num`          | Use `1` for single dataset, `2` for multi-dataset continual learning. |
| `dataset` / `dataset2` | Dataset identifiers (e.g., "C100", "C10", "STL10"). |
| `training_mode`     | Strategy for learning tasks (e.g., 'consecutive'). |
| `top_k`             | Fraction of top kernels to protect from overwriting. |
| `topk_lock`         | Boolean to freeze top-k kernel weights. |
| `high_lr` / `low_lr`| Learning rate modifiers for plastic vs important kernels. |
| `t_criteria`        | Importance criterion: 'activations' or 'KSE'. |
| `delta_w_interval`  | Interval (in batches) for tracking kernel updates. |

Modify these parameters directly in `t_hyper.py` to tailor your experimental design.
To run the experiments after setting the right hyperparameters in `t_hyper.py`:
```bash
cd batches
python testing.py
```
This is going to create independent batch jobs to run the experiments.

---
