# Incremental Learning for Audio Classification with Hebbian Deep Neural Networks

## Author

Riccardo Casciotti, Tampere University  
Co-authors: Prof. Alberto Antonietti, Politecnico di Milano; Francesco De Santis, Politecnico di Milano; Prof. Annamaria Mesaros, Tampere University

## Overview

This repository contains experiments for **task-incremental audio classification** with a SoftHebb-inspired neural architecture. The code trains a Hebbian convolutional feature extractor together with supervised classifier heads, then evaluates how performance changes as new tasks/classes are introduced.

The main continual-learning mechanisms are:

- **Kernel plasticity / CF solution**: ranks kernels by importance and changes their learning rate during later tasks to reduce catastrophic forgetting.
- **Multi-head / head solution**: stores a separate classifier head for each task and reloads the correct head during evaluation.
- **Joint baseline inside each run**: after each new task, the code also trains/evaluates a joint model over all classes seen so far.
- **LwF/EWC baselines**: implemented separately in `SoftHebb-main/baselines_LwF_EWC.py`.

Supported experiment datasets in the current code path are **ESC-50** and **UrbanSound8K**.

---

## Repository structure

```text
Hebbian-TIL-develop/
├── README.md
├── SoftHebb-main/
│   ├── continual_learning.py      # Main continual-learning experiment runner
│   ├── baselines_LwF_EWC.py       # LwF and EWC baseline runner
│   ├── presets.json               # Model, layer, and dataset presets
│   ├── dataset.py                 # ESC-50 / UrbanSound8K dataset loaders
│   ├── model.py                   # SoftHebb model construction and checkpoint loading
│   ├── engine_cl.py               # Continual-learning training/evaluation loops
│   ├── hebbconv.py                # Hebbian convolution layer and plasticity masks
│   └── train.py                   # Supervised/unsupervised training utilities
├── experiment_utils/
│   ├── testing.py                 # Generates many experiment submission commands
│   ├── configs.sh                 # Generated/ready-to-run batch commands
│   └── execs/
│       ├── ESC50.sh               # SLURM wrapper for ESC-50 experiments
│       ├── URBANSOUND8K.sh        # SLURM wrapper for UrbanSound8K experiments
│       ├── ESC50_apple.sh         # Local Apple Silicon wrapper, path-specific
│       └── baselines.sh           # SLURM wrapper for LwF/EWC baselines
└── experiment_results/            # Example/result-processing files from previous runs
```

---

## Before running: environment, paths, and data

### 1. Create a Python environment

The repository does not currently include a requirements file. The code imports at least the following packages:

```bash
conda create -n softhebb python=3.8 -y
conda activate softhebb
pip install numpy pandas h5py matplotlib seaborn torch torchvision torchaudio
```

On CSC/SLURM, the provided job scripts expect a Singularity image at:

```text
/scratch/project_462001198/casciott/softhebb_env/softhebb.sif
```

If you are not running on that exact environment, update the paths in the shell scripts under `experiment_utils/execs/`.

### 2. Check hard-coded paths

Several files contain machine-specific paths. Before launching a run on a different machine or project directory, update these paths:

| File | What to check |
|---|---|
| `SoftHebb-main/utils.py` | `BASE_PATH`, `DATA`, `DATASET`, `RESULT`, and the hard-coded path used to open `presets.json`. |
| `SoftHebb-main/dataset.py` | ESC-50 path under `Training/data/ESC-50-master`; UrbanSound8K HDF5 path under `/scratch/.../datasets/urbansound8k`. |
| `SoftHebb-main/continual_learning.py` | `BASE_PATH` used for result saving and cleanup. |
| `experiment_utils/testing.py` | `BASE_PATH`, `parent_f_id`, and the path where `configs.sh` is written. |
| `experiment_utils/execs/*.sh` | SLURM account/partition, container path, bind mounts, and Python script paths. |

### 3. Prepare the datasets

The current loaders expect:

| Dataset | Expected format/path in the current code |
|---|---|
| ESC-50 | Raw ESC-50 folder with metadata at `<BASE_PATH>/Training/data/ESC-50-master/meta/esc50.csv` and audio under `<BASE_PATH>/Training/data/ESC-50-master/audio/`. |
| UrbanSound8K | HDF5 file at `/scratch/project_462001198/casciott/datasets/urbansound8k/h5s/urbansound8k.h5`. |
| ESC-50 baselines | HDF5 file at `<data_path>/h5s/esc50.h5`, where `data_path` defaults to `/scratch/project_462001198/casciott/datasets/esc50`. |

---

## Quick start: run an experiment

There are two ways to start an experiment.

### Option A — Generate and submit several SLURM jobs

This is the workflow used by `experiment_utils/testing.py`.

1. Open `experiment_utils/testing.py`.
2. Choose the dataset and experiment size:

```python
TEST = False          # True = reduced/debug run
SHMH = False          # Special single-head/multi-head behavior; keep False unless needed
SINGLE = False        # True = train one task containing all classes
n_experiments = 2     # Number of random task splits / repeated runs

dataset = "ESC50"     # Options: "ESC50" or "URBANSOUND8K"
```

3. Set the task structure:

```python
# ESC-50 default: task 0 has 30 classes, tasks 1-4 have 5 classes each.
classes_per_task = 5
n_tasks = 5

evaluated_tasks = [0, 1, 2, 3, 4]
```

4. Set the continual-learning hyperparameters in `cl_hyper`:

```python
cl_hyper = {
    "training_mode": "consecutive",
    "top_k": 0.6,
    "topk_lock": False,
    "high_lr": 0.15,
    "low_lr": 0.9,
    "t_criteria": "activations",
    "delta_w_interval": 5,
    "heads_basis_t": 0.90,
    "n_tasks": n_tasks,
    "classes_per_task": classes_per_task,
}
```

5. Generate the batch commands:

```bash
cd Hebbian-TIL-develop
python experiment_utils/testing.py
```

This writes one `sbatch ...` command per run into `experiment_utils/configs.sh` or into the path configured inside `testing.py`.

6. Submit the jobs:

```bash
bash experiment_utils/configs.sh
```

7. Follow the logs:

```bash
tail -f experiment_utils/execs/ESC50/job.out
tail -f experiment_utils/execs/ESC50/job.err
```

For UrbanSound8K, use the corresponding `URBANSOUND8K/job.out` and `URBANSOUND8K/job.err` paths.

> Important: `continual_learning.py` defines the argument as `--evaluated-tasks` with an **s**. If a wrapper script uses `--evaluated-task`, change it to `--evaluated-tasks` before running.

### Option B — Run one experiment directly

Use this when debugging one configuration without generating many SLURM jobs.

From the repository root:

```bash
cd SoftHebb-main
python continual_learning.py \
  --preset 6SoftHebbCnnESC50 \
  --resume all \
  --model-name ESC50_CL \
  --dataset-unsup ESC50_1 \
  --dataset-sup ESC50_100 \
  --continual_learning True \
  --evaluate True \
  --training-mode consecutive \
  --cf-sol True \
  --head-sol True \
  --top-k 0.6 \
  --high-lr 0.15 \
  --low-lr 0.9 \
  --t-criteria activations \
  --delta-w-interval 5 \
  --heads-basis-t 0.90 \
  --selected-classes '[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],[30,31,32,33,34],[35,36,37,38,39],[40,41,42,43,44],[45,46,47,48,49]]' \
  --n-tasks 5 \
  --evaluated-tasks '[0,1,2,3,4]' \
  --classes-per-task 5 \
  --topk-lock False \
  --folder-id _debug5tasks \
  --parent-f-id experiments/EXP_ESC50_5C \
  --shmh False
```

For UrbanSound8K, change the preset and dataset fields:

```bash
--preset 6SoftHebbCnnUrbanSound8k \
--model-name URBANSOUND8K_CL \
--dataset-unsup URBANSOUND8K_1 \
--dataset-sup URBANSOUND8K_100
```

---

## What happens during a continual-learning run

For every fold and every task, `continual_learning.py` performs the following steps:

1. Loads the model preset from `SoftHebb-main/presets.json`.
2. Creates dataset configurations from the selected dataset preset.
3. Trains task 0 from scratch.
4. Evaluates task 0.
5. For each later task:
   - reloads the previous checkpoint,
   - trains on the new task,
   - applies the kernel plasticity solution if `cf_sol=True`,
   - stores/uses task heads if `head_sol=True`,
   - evaluates all requested previous/current tasks,
   - trains a joint model over classes seen so far for comparison.
6. Saves a JSON result file under:

```text
<BASE_PATH>/<parent_f_id>/TASKS_CL_<dataset><folder_id>/<model_name>.json
```

The code uses 5 folds for ESC-50 and 10 folds for UrbanSound8K.

---

## Meaning of fields in `experiment_utils/testing.py`

| Field | Meaning |
|---|---|
| `TEST` | Debug switch. When `True`, the script reduces the run size and only generates one solution/run. |
| `SHMH` | Special single-head/multi-head flag passed to `--shmh`. Keep `False` for the standard multi-head flow. The current evaluation code warns that `POP_HEAD=True` and `SHMH=True` are not supported together. |
| `SINGLE` | If `True`, creates one task containing all classes instead of an incremental sequence. Useful for non-continual single-task testing. |
| `n_experiments` | Number of independent class orderings/random task splits to generate. Each split becomes one or more `sbatch` commands. |
| `dataset` | Dataset name used by the batch generator. Supported values in this file: `"ESC50"`, `"URBANSOUND8K"`. |
| `classes_per_task` | Number of classes in each incremental task. For ESC-50, the current code uses a special split: 30 classes in task 0, then 5 classes per later task. |
| `n_tasks` | Number of tasks in the incremental sequence. Default ESC-50 and UrbanSound8K settings use 5. |
| `evaluated_tasks` | Tasks to evaluate after training each task. If the list contains indices beyond `n_tasks - 1`, the runner clips it to `range(n_tasks)`. |
| `id` | Free-text experiment identifier used to build `folder_id`. |
| `folder_id` | Suffix added to the result folder name. Example: `_full_run_gitcode5tasks`. |
| `parent_f_id` | Parent result directory relative to `BASE_PATH`. Example: `experiments/EXP_ESC50_5C`. |
| `BASE_PATH` | Root path where experiments and results are stored. Currently cluster-specific. |
| `command` | Base SLURM command. It changes into `experiment_utils/execs/` and calls `sbatch <dataset>.sh ...`. |
| `all_classes` | List of global class labels used to create task splits. ESC-50 uses `0..49`; UrbanSound8K uses `0..9`. |
| `classes` / `final` | Generated list of task splits. This becomes the `--selected-classes` argument. |
| `sols` | Solution combinations to run. Each tuple is `(cf_sol, head_sol)`. For standard ESC-50 with `SINGLE=False`, the script generates `(False, True)` and `(True, True)`. |

---

## Meaning of `cl_hyper` fields

`cl_hyper` is created in `testing.py`, passed through the shell scripts, and reconstructed inside `continual_learning.py`.

| Field | Meaning |
|---|---|
| `training_mode` | Controls the order of block training. Accepted values are `successive`, `consecutive`, and `simultaneous`. The default experiments use `consecutive`. |
| `cf_sol` | Enables the catastrophic-forgetting / kernel-plasticity solution. When `True`, the code tracks activations and weight changes, ranks kernels, and modifies Hebbian learning rates. |
| `head_sol` | Enables the multi-head solution. When `True`, the model keeps task-specific classifier heads and reloads the appropriate head for evaluation. |
| `top_k` | Fraction of kernels considered important according to `t_criteria`. Example: `0.6` protects/marks roughly the top 60% of kernels per layer. |
| `topk_lock` | If `True`, the top-k kernel mask is used directly to reduce/freeze protected kernels, instead of combining it with the weight-change threshold mask. |
| `high_lr` | Relative increase applied to non-top-k kernels. Example: `0.15` makes their update multiplier approximately `1.15`. |
| `low_lr` | Relative decrease applied to protected kernels. Example: `0.9` makes their update multiplier approximately `0.10` when the lower-learning-rate mask is active. |
| `t_criteria` | Kernel ranking criterion. `activations` ranks kernels by accumulated activation magnitude. `KSE` is referenced in the model-loading path but the default current experiments use `activations`. |
| `delta_w_interval` | Number of training iterations between weight-change measurements. Smaller values update the tracked deltas more often. |
| `heads_basis_t` | Threshold value saved in the model as `heads_thresh`. It is intended for head-selection/head-basis logic and is stored in checkpoints. |
| `n_tasks` | Number of tasks in the run. |
| `classes_per_task` | Expected number of classes per incremental task. For ESC-50, task 0 is currently a special larger task. |
| `selected_classes` | List of class lists, one list per task. Example: `[[0,1,2],[3,4],[5,6]]`. The shell scripts pass this as a quoted JSON/Python list string. |
| `evaluated_tasks` | Task indices evaluated after each training phase. |
| `shmh` | Special mode flag copied into the dataset/model config. Leave `False` for normal task-incremental multi-head experiments. |
| `SINGLE` | Internal flag indicating single-task mode. |

---

## Meaning of `continual_learning.py` command-line arguments

| Argument | Meaning |
|---|---|
| `--preset` | Model preset key from `presets.json`, such as `6SoftHebbCnnESC50` or `6SoftHebbCnnUrbanSound8k`. |
| `--folder-id` | Experiment folder suffix used when saving results. |
| `--parent-f-id` | Parent results folder. |
| `--dataset-unsup` | Dataset preset used for unsupervised/Hebbian blocks. Examples: `ESC50_1`, `URBANSOUND8K_1`. The suffix selects a dataset sub-preset in `presets.json`; `_1` means 1 epoch. |
| `--dataset-sup` | Dataset preset used for supervised classifier training. Examples: `ESC50_100`, `URBANSOUND8K_100`. |
| `--dataset-unsup-1`, `--dataset-sup-1` | Legacy/extra dataset arguments parsed by the script but not used in the current main execution path. |
| `--continual_learning` | Boolean flag copied into the dataset config. During the scripted task loop, the code sets this internally for each task phase. |
| `--training-mode` | Training schedule: `successive`, `consecutive`, or `simultaneous`. |
| `--resume` | Checkpoint loading mode. `all` loads the previous full model; `without_classifier` is available in the parser; `None` starts from scratch. |
| `--model-name` | Prefix for saved checkpoint/result names. A UUID is appended automatically. |
| `--training-blocks` | Optional list of block indices to train. If omitted, the training configuration decides the blocks from the preset. |
| `--seed` | Random seed copied into dataset configs. |
| `--gpu-id` | GPU index argument. The current `get_device()` implementation selects CUDA if available. |
| `--save` | Enables/disables checkpoint saving. |
| `--validation` | Enables validation split handling in dataset config. |
| `--evaluate` | Evaluation flag. The main task loop also sets evaluation internally. |
| `--topk-lock` | Enables direct locking/reduction of top-k kernels. |
| `--skip-1` | Legacy flag intended to skip first-dataset training and resume directly; not used in the current main loop. |
| `--classes-per-task` | Number of classes per task used to initialize dataset output sizes. |
| `--head-sol` | Enables/disables task-specific heads. |
| `--cf-sol` | Enables/disables the kernel-plasticity continual-learning solution. |
| `--top-k` | Fraction of kernels selected as important. |
| `--high-lr` | Relative learning-rate increase for non-protected kernels. |
| `--low-lr` | Relative learning-rate decrease for protected kernels. |
| `--delta-w-interval` | Interval for measuring weight deltas. |
| `--t-criteria` | Kernel ranking criterion, usually `activations`. |
| `--heads-basis-t` | Head threshold value stored in checkpoints. |
| `--selected-classes` | String representation of the tasks/classes list. The script evaluates this string, so keep it quoted. |
| `--n-tasks` | Number of tasks. |
| `--evaluated-tasks` | String representation of the list of task indices to evaluate. Keep the plural form. |
| `--shmh` | Special head-mode flag; leave `False` for standard experiments. |

---

## Positional fields passed by the SLURM wrappers

`testing.py` writes commands like:

```bash
sbatch ESC50.sh <1> <2> ... <17>
```

The wrapper scripts map these positions to `continual_learning.py` arguments as follows:

| Position | Shell variable | Meaning |
|---:|---|---|
| 1 | `$1` | `training_mode` |
| 2 | `$2` | `cf_sol` |
| 3 | `$3` | `head_sol` |
| 4 | `$4` | `top_k` |
| 5 | `$5` | `high_lr` |
| 6 | `$6` | `low_lr` |
| 7 | `$7` | `t_criteria` |
| 8 | `$8` | `delta_w_interval` |
| 9 | `$9` | `heads_basis_t` |
| 10 | `${10}` | `selected_classes` |
| 11 | `${11}` | `n_tasks` |
| 12 | `${12}` | `evaluated_tasks` |
| 13 | `${13}` | `classes_per_task` |
| 14 | `${14}` | `topk_lock` |
| 15 | `${15}` | `folder_id` |
| 16 | `${16}` | `parent_f_id` |
| 17 | `${17}` | `shmh` |

---

## Meaning of important `presets.json` fields

`presets.json` has three top-level sections: `model`, `layer`, and `dataset`.

### `model`

A model preset is a sequence of blocks, for example `6SoftHebbCnnESC50`.

| Field | Meaning |
|---|---|
| `b0`, `b1`, ... | Ordered model blocks. Earlier blocks are convolutional Hebbian blocks; the final block is usually an MLP classifier. |
| `arch` | Block type: `CNN` or `MLP`. |
| `preset` | Compact layer description. Example: `softkrotov-c48-k5-p2-s1-d1-b0-t1.1-lr0.08`. |
| `operation` | Operation wrapped around the layer, such as `batchnorm2d` or `flatten`. |
| `num` | Block index. |
| `batch_norm` | Whether batch normalization is enabled for that block. |
| `pool` | Pooling specification in the form `<type>_<kernel_size>_<stride>_<padding>`, such as `max_4_2_1`. |
| `activation` | Activation specification, such as `triangle_0.7`. |
| `resume` | Optional resume/checkpoint behavior for that block. |
| `dropout`, `att_dropout` | Dropout settings used by MLP/classifier blocks. |

### Compact layer preset tokens

| Token | Meaning |
|---|---|
| `softkrotov` | Hebbian/SoftHebb-style layer. |
| `BP` | Backpropagation-trained layer. |
| `c<number>` | Output channels/classes/neurons, depending on layer type. |
| `k<number>` | CNN kernel size. |
| `p<number>` | CNN padding. |
| `s<number>` | CNN stride. |
| `d<number>` | CNN dilation. |
| `b0` / `b1` | Disable/enable bias. |
| `t<number>` | `t_invert`, the SoftHebb activation temperature/inversion factor. |
| `lr<number>` | Hebbian learning rate. |
| `ls<number>` | Supervised learning rate. |
| `lb<number>` | Lebesgue-p value. |
| `lp<number>` | Power learning-rate value. |
| `a<number>` | Delta parameter. |
| `r<number>` | Radius parameter. |
| `v0` / `v1` | Disable/enable adaptive behavior. |

### `layer`

Layer defaults define values used when a compact preset does not override them.

| Field | Meaning |
|---|---|
| `lr` | Hebbian learning rate. |
| `lr_sup` | Supervised learning rate. |
| `adaptive` | Enables adaptive behavior in the layer. |
| `speed`, `lr_div`, `lr_decay` | Learning-rate schedule controls. |
| `lebesgue_p` | Norm/order parameter used in the layer computations. |
| `t_invert` | SoftHebb activation scaling/temperature parameter. |
| `beta`, `power`, `power_lr` | Activation and update-shaping parameters. |
| `ranking_param` | Ranking parameter used by Hebbian layer logic. |
| `delta` | Delta/update control parameter. |
| `hebbian` | Whether the layer uses Hebbian learning. |
| `add_bias` | Whether to add bias parameters. |
| `normalize_inp` | Whether to normalize inputs. |
| `softness`, `soft_activation_fn` | SoftHebb activation behavior. |
| `plasticity` | Plasticity rule name. |
| `metric_mode` | Metric mode, for example `unsupervised`. |
| `weight_init`, `weight_init_range`, `weight_init_offset` | Weight initialization configuration. |
| `weight_decay` | Weight decay coefficient. |
| `radius` | Radius parameter used by Hebbian layers. |
| `padding_mode`, `padding`, `stride`, `dilation`, `groups` | CNN convolution settings. |
| `mask_thsd` | Mask threshold. |
| `pre_triangle` | Whether to apply pre-triangle behavior in CNN config. |
| `seed` | Layer-level random seed. |

### `dataset`

Dataset presets configure input shape, sample counts, and training duration.

| Field | Meaning |
|---|---|
| `name` | Dataset name used by `dataset.py`, such as `ESC50` or `URBANSOUND8K`. |
| `split` | Dataset split name used by the preset. |
| `channels` | Number of input channels. Audio spectrograms use `1`. |
| `width`, `height` | Expected input dimensions after preprocessing. |
| `n_mels` | Number of mel bands for spectrograms. |
| `n_fft` | FFT window size for mel-spectrogram extraction. |
| `hop_len` | Hop length for mel-spectrogram extraction. |
| `out_channels` | Total number of dataset classes before task splitting. |
| `training_sample`, `testing_sample` | Dataset sample counts used in configuration. |
| `nb_epoch` | Number of epochs for the selected dataset preset. For example, `ESC50_1` overrides this to 1 epoch. |
| `print_freq` | How often training progress is printed. |
| `batch_size` | Batch size. |
| `num_workers` | DataLoader worker count. |
| `seed` | Dataset seed. |
| `shuffle` | Whether to shuffle data where applicable. |
| `augmentation` | Whether data augmentation is enabled. |
| `zca_whitened` | Whether ZCA whitening is expected/enabled. |
| `validation_split` | Fraction of training data used for validation when validation is enabled. |
| `training_class` | Class selection field used by older/general dataset logic. |

---

## Meaning of result JSON fields

Each experiment writes a JSON file containing the training/evaluation summary.

| Field | Meaning |
|---|---|
| `FOLD_#1`, `FOLD_#2`, ... | Per-fold task results. Each fold contains entries like `T0`, `T1`, and evaluation summaries such as `eval_0`. |
| `accuracy_matrix` | Accuracy after each training step on each evaluated task. This is the main structure for measuring forgetting. |
| `joint` | Joint-training baseline results over the classes seen so far. |
| `confusion_matrix` | Confusion matrices from multi-head evaluation, organized by fold and task. |
| `performance_avg_folds` | Average task performance across folds. |
| `model_config` | Expanded model/block configuration used for the run. |
| `model_name` | Final model name, including the generated UUID suffix. |
| `count` | Internal counter used while filling the result object. |

---

## Running LwF/EWC baselines

The baseline script is independent from the SoftHebb continual-learning runner:

```bash
cd SoftHebb-main
python baselines_LwF_EWC.py \
  --data_path /scratch/project_462001198/casciott/datasets/esc50 \
  --strategy both \
  --train_epochs 100 \
  --lr 1e-3 \
  --tasks '[[34,12,11,45,31,14,28,13,25,49,46,33,15,39,29,3,47,4,44,36,35,7,23,21,1,40,0,9,41,32],[5,26,27,16,38],[43,10,24,20,18],[37,17,22,19,30],[6,42,48,2,8]]'
```

Important baseline arguments:

| Argument | Meaning |
|---|---|
| `--data_path` | Folder containing `h5s/esc50.h5`. |
| `--strategy` | `lwf`, `ewc`, or `both`. |
| `--device` | Device string, for example `cuda` or `cpu`. |
| `--seed` | Random seed. |
| `--debug` | Uses a smaller debug subset. |
| `--stats_num_classes` | Global number of classes, normally `50` for ESC-50. |
| `--train_mb_size`, `--eval_mb_size` | Training and evaluation minibatch sizes. |
| `--train_epochs` | Number of supervised training epochs per task. |
| `--lr` | Optimizer learning rate. |
| `--lwf_alpha` | LwF distillation loss weight. |
| `--lwf_temperature` | LwF distillation temperature. |
| `--ewc_lambda` | EWC regularization strength. |
| `--ewc_mode` | EWC accumulation mode: `separate`, `onlinesum`, or `onlineweightedsum`. |
| `--log_class_hists` | Logs class histograms. |
| `--tasks` | JSON list of task class lists. |

---

## Common pitfalls

- **Wrong path errors**: most failures on new machines come from hard-coded `/projappl/...`, `/scratch/...`, or `/Users/...` paths.
- **Argument typo**: use `--evaluated-tasks`, not `--evaluated-task`.
- **Quoted lists**: keep `--selected-classes` and `--evaluated-tasks` inside quotes so the shell passes them as one argument.
- **ESC-50 split**: the default ESC-50 experiment is not equal-sized across all tasks; task 0 uses 30 classes, followed by four 5-class tasks.
- **`SHMH=True`**: the current evaluation path prints a warning when `POP_HEAD=True` and `SHMH=True`, so use `SHMH=False` for standard runs.
- **Result overwrite**: choose a new `id`/`folder_id` for each experiment if you want a separate result directory.