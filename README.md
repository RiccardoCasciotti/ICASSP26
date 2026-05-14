# Incremental Learning for Audio Classification with Hebbian Deep Neural Networks

## Author

Riccardo Casciotti, Tampere University  
Co-authors: Prof. Alberto Antonietti, Politecnico di Milano; Francesco De Santis, Politecnico di Milano; Prof. Annamaria Mesaros, Tampere University

## Overview

This repository contains task-incremental audio-classification experiments based on a SoftHebb-inspired convolutional architecture. The main workflow is controlled by:

```text
experiment_utils/experiment_config.json
```

A user normally edits this JSON file, then launches `experiment_utils/testing.py`. The script reads the JSON, creates random task/class splits, chooses the continual-learning solution combinations, and submits one or more SLURM jobs with `sbatch`.

Supported datasets in the current experiment launcher are:

- `ESC50`
- `URBANSOUND8K`

The main continual-learning options are:

- `cf_sol`: kernel-plasticity / catastrophic-forgetting solution.
- `head_sol`: multi-head classifier solution.
- `top_k`, `high_lr`, `low_lr`, `t_criteria`, `delta_w_interval`: hyperparameters controlling how kernels are selected and how their learning rates are modified.

---

## Repository structure

```text
Hebbian-TIL-develop/
├── README.md
├── SoftHebb-main/
│   ├── continual_learning.py      # Main continual-learning runner
│   ├── baselines_LwF_EWC.py       # LwF/EWC baseline runner
│   ├── presets.json               # Model, layer, and dataset presets
│   ├── dataset.py                 # ESC-50 and UrbanSound8K dataset loading
│   ├── model.py                   # Model construction/checkpoint loading
│   ├── engine_cl.py               # Continual-learning train/eval loops
│   └── hebbconv.py                # Hebbian convolution layer and plasticity logic
├── experiment_utils/
│   ├── experiment_config.json     # Main file users edit to start experiments
│   ├── testing.py                 # Reads the JSON and submits sbatch jobs
│   ├── configs.sh                 # Older/generated command file; not the main current workflow
│   └── execs/
│       ├── ESC50.sh               # SLURM wrapper for ESC-50
│       ├── URBANSOUND8K.sh        # SLURM wrapper for UrbanSound8K
│       ├── ESC50_apple.sh         # Local Apple-specific wrapper
│       └── baselines.sh           # SLURM wrapper for LwF/EWC baselines
└── experiment_results/            # Example/result-processing files from previous runs
```

---

## Quick start: run an experiment from `experiment_config.json`

### 0. Create and Activate the Conda Environment

```bash
conda env create -f softhebb.yml
conda activate softhebb_env
pip install -r softhebb_env/pip_reqs.txt
```


### 1. Edit the experiment configuration

Open:

```bash
experiment_utils/experiment_config.json
```

A minimal example for an UrbanSound8K debug run is:

```json
{
  "debug": true,
  "single": false,
  "n_experiments": 1,
  "dataset": "URBANSOUND8K",
  "id": "TEST",

  "base_path": "/scratch/project_462001198/casciott",
  "project_path": "/projappl/project_462001198/casciott/ICASSP26",
  "presets_path": "/projappl/project_462001198/casciott/ICASSP26/SoftHebb-main/presets.json",
  "results_path": "/scratch/project_462001198/casciott/Training/results/hebb/result",
  "preset_name": "6SoftHebbCnnUrbanSound8k",
  "datasets_path": "/scratch/project_462001198/casciott/datasets/urbansound8k",
  "execs_dir": "/projappl/project_462001198/casciott/ICASSP26/experiment_utils/execs",

  "evaluated_tasks": [0, 1, 2, 3, 4],

  "cl_hyper": {
    "training_mode": "consecutive",
    "top_k": 0.6,
    "topk_lock": false,
    "high_lr": 0.15,
    "low_lr": 0.9,
    "t_criteria": "activations",
    "delta_w_interval": 5
  }
}
```

For ESC-50, the most important changes are:

```json
{
  "dataset": "ESC50",
  "preset_name": "6SoftHebbCnnESC50",
  "datasets_path": "/path/to/ESC-50-master"
}
```

The current default `experiment_config.json` may have `dataset` set to `URBANSOUND8K` while `preset_name` is set to `6SoftHebbCnnESC50`. Before running, make sure these two fields match.

### 2. Check the SLURM wrappers once

The launcher submits one of these scripts depending on the selected dataset:

```text
experiment_utils/execs/ESC50.sh
experiment_utils/execs/URBANSOUND8K.sh
```

Update these scripts if your cluster account, partition, container path, bind mounts, or project paths are different.

Also check the evaluated-task argument name. `SoftHebb-main/continual_learning.py` expects:

```bash
--evaluated-tasks
```

If a wrapper contains `--evaluated-task`, change it to `--evaluated-tasks` before launching.

### 3. Launch the experiment

From the repository root:

```bash
python experiment_utils/testing.py
```

When prompted:

```text
Insert the full path for the experiment configuration json file:
```

paste the full path to your configuration file, for example:

```text
/projappl/project_462001198/casciott/ICASSP26/experiment_utils/experiment_config.json
```

`testing.py` then:

1. reads the JSON file,
2. infers the number of tasks and classes per task from `dataset` and `single`,
3. randomly generates the class split for each experiment,
4. derives the result folders from `id`, `dataset`, and the task structure,
5. chooses the solution combinations to run,
6. submits one `sbatch` job for each generated experiment/solution pair.

### 4. Read job logs

For ESC-50:

```bash
tail -f experiment_utils/execs/ESC50/job.out
tail -f experiment_utils/execs/ESC50/job.err
```

For UrbanSound8K:

```bash
tail -f experiment_utils/execs/URBANSOUND8K/job.out
tail -f experiment_utils/execs/URBANSOUND8K/job.err
```

### 5. Find results

The launcher creates/checks folders under:

```text
<base_path>/<parent_f_id>/TASKS_CL_<dataset><folder_id>/
```

where `parent_f_id` and `folder_id` are generated automatically:

```python
folder_id = f"_{id}{n_tasks}tasks"
parent_f_id = f"experiments/EXP_{dataset}_{classes_per_task}C"
```

Example for UrbanSound8K with `id = "TEST"`, `n_tasks = 5`, and `classes_per_task = 2`:

```text
/scratch/project_462001198/casciott/experiments/EXP_URBANSOUND8K_2C/TASKS_CL_URBANSOUND8K_TEST5tasks/
```

---

## Meaning of every field in `experiment_config.json`

### Top-level fields

| Field | Type | Used by | Meaning |
|---|---:|---|---|
| `debug` | boolean | `testing.py` | Enables a reduced/debug run. When `true`, `testing.py` forces `n_experiments = 1`, uses `n_tasks = 2`, and only runs the `(cf_sol=True, head_sol=True)` solution. Use `true` for a quick test before launching a full run. Use `false` for the real experiment. |
| `single` | boolean | `testing.py` | If `true`, all classes are placed into one task. This disables the normal incremental sequence. If `false`, classes are split into multiple tasks. |
| `n_experiments` | integer | `testing.py` | Number of independent random class splits to generate. Each split can produce multiple SLURM jobs because `testing.py` may run more than one solution combination. Ignored/reduced to `1` when `debug` is `true`. |
| `dataset` | string | `testing.py` and wrapper selection | Dataset to run. Supported values are `"ESC50"` and `"URBANSOUND8K"`. This determines the SLURM wrapper, number of classes, classes per task, and task split logic. |
| `_comment` | string | not used | Human-readable comment. It has no effect on the code. |
| `id` | string | `testing.py` | Experiment label used to create the output folder suffix. For example, `"TEST"` with 5 tasks becomes `folder_id = "_TEST5tasks"`. Change this for each meaningful run to avoid mixing results. |
| `base_path` | string/path | `testing.py` | Root path where `testing.py` creates/checks experiment result folders. The script creates `<base_path>/experiments` and `<base_path>/<parent_f_id>` if needed. |
| `project_path` | string/path | mostly documentation in current code | Path to the repository/project on the cluster. The current `testing.py` does not directly read this field when building commands, but the shell wrappers contain hard-coded project paths that should usually match this value. |
| `presets_path` | string/path | passed to `continual_learning.py` | Full path to `SoftHebb-main/presets.json`. The runner uses this file to load model, layer, and dataset presets. |
| `results_path` | string/path | passed to `continual_learning.py` | Base path used by the model/checkpoint/result logic inside the training code. This is passed to `--results_path`. |
| `preset_name` | string | passed to `continual_learning.py` | Model preset key inside `presets.json`. Use `6SoftHebbCnnESC50` for ESC-50 and `6SoftHebbCnnUrbanSound8k` for UrbanSound8K unless you add new presets. |
| `datasets_path` | string/path | passed to `dataset.py` through `continual_learning.py` | Dataset root path. For ESC-50 this should point to the `ESC-50-master` folder containing `meta/esc50.csv` and `audio/`. For UrbanSound8K this should point to the folder containing `h5s/urbansound8k.h5`. |
| `execs_dir` | string/path | `testing.py` | Folder containing the SLURM wrapper scripts. `testing.py` runs `sbatch <DATASET>.sh` with `cwd=execs_dir`, so this directory must contain `ESC50.sh` and/or `URBANSOUND8K.sh`. |
| `evaluated_tasks` | list of integers | passed to `continual_learning.py` | Task indices evaluated after training. For a normal 5-task run use `[0, 1, 2, 3, 4]`. If this list contains indices greater than `n_tasks - 1`, `continual_learning.py` clips it to all valid tasks. |
| `cl_hyper` | object/dict | copied and extended by `testing.py` | Continual-learning hyperparameters. See the next table. |

### `cl_hyper` fields

| Field | Type | Meaning |
|---|---:|---|
| `training_mode` | string | Training schedule passed to `--training-mode`. Accepted values in `continual_learning.py` are `successive`, `consecutive`, and `simultaneous`. The standard experiments use `consecutive`. |
| `top_k` | float | Fraction of kernels selected as important according to `t_criteria`. Example: `0.6` selects roughly the top 60% of kernels per Hebbian convolutional layer. |
| `topk_lock` | boolean | Controls how the top-k kernel mask is applied. When `false`, the code combines top-k information with the weight-change threshold logic. When `true`, the top-k mask is used more directly to reduce/lock protected kernels. The standard setting is `false`. |
| `high_lr` | float | Relative learning-rate increase for non-protected kernels. Example: `0.15` makes the multiplier approximately `1.15` for kernels selected for increased plasticity. |
| `low_lr` | float | Relative learning-rate decrease for protected kernels. Example: `0.9` makes the multiplier approximately `0.10` where the lower-learning-rate mask is active. Higher values mean stronger reduction. |
| `t_criteria` | string | Criterion for choosing important kernels. The recommended/default experiment value is `activations`, which ranks kernels by accumulated activation magnitude. `KSE` is referenced in the model code, but the current configuration uses `activations`. |
| `delta_w_interval` | integer/float | Number of training iterations between weight-change measurements. Smaller values update the tracked deltas more frequently and add overhead. Larger values measure less often. |

### Fields added automatically by `testing.py`

These fields are not written manually in `experiment_config.json`; `testing.py` derives them and passes them to the shell wrapper / training script.

| Generated field | Meaning |
|---|---|
| `n_tasks` | Number of tasks. For normal runs, both datasets use 5 tasks. If `single=true`, this becomes 1. If `debug=true`, this becomes 2 inside `testing.py`. |
| `classes_per_task` | Number of classes per incremental task. UrbanSound8K uses 2. ESC-50 uses 5 for incremental tasks, but its current split is special: task 0 has 30 classes and tasks 1-4 have 5 classes each. |
| `selected_classes` | Randomly generated list of class lists, one list per task. This is what determines which classes belong to each task in a run. |
| `folder_id` | Result-folder suffix generated from `id` and `n_tasks`, for example `_TEST5tasks`. |
| `parent_f_id` | Parent experiment folder generated from dataset and class count, for example `experiments/EXP_ESC50_5C`. |
| `cf_sol` | Whether to enable the kernel-plasticity catastrophic-forgetting solution. This is not currently controlled from JSON; it is selected by `get_solutions()` in `testing.py`. |
| `head_sol` | Whether to enable the multi-head classifier solution. This is not currently controlled from JSON; it is selected by `get_solutions()` in `testing.py`. |
| `SINGLE` | Internal copy of `single`, stored in `cl_hyper`. |

---

## Dataset-specific behavior

### ESC-50

`testing.py` uses:

```python
classes_per_task = 5
all_classes = list(range(50))
```

For normal incremental ESC-50 runs, the split is:

```text
Task 0: 30 classes
Task 1: 5 classes
Task 2: 5 classes
Task 3: 5 classes
Task 4: 5 classes
```

So even though `classes_per_task = 5`, task 0 is intentionally larger in the current code.

Recommended matching fields:

```json
{
  "dataset": "ESC50",
  "preset_name": "6SoftHebbCnnESC50",
  "datasets_path": "/path/to/ESC-50-master",
  "evaluated_tasks": [0, 1, 2, 3, 4]
}
```

Expected ESC-50 dataset structure:

```text
ESC-50-master/
├── meta/esc50.csv
└── audio/
```

### UrbanSound8K

`testing.py` uses:

```python
classes_per_task = 2
all_classes = list(range(10))
```

For normal incremental UrbanSound8K runs, the split is:

```text
Task 0: 2 classes
Task 1: 2 classes
Task 2: 2 classes
Task 3: 2 classes
Task 4: 2 classes
```

Recommended matching fields:

```json
{
  "dataset": "URBANSOUND8K",
  "preset_name": "6SoftHebbCnnUrbanSound8k",
  "datasets_path": "/path/to/urbansound8k",
  "evaluated_tasks": [0, 1, 2, 3, 4]
}
```

Expected UrbanSound8K dataset structure:

```text
urbansound8k/
└── h5s/urbansound8k.h5
```

---

## Which jobs are submitted?

`testing.py` does not currently let the JSON directly set `cf_sol` and `head_sol`. Instead, it calls `get_solutions(dataset, single, test)`.

The current solution combinations are:

| Condition | Submitted `(cf_sol, head_sol)` combinations |
|---|---|
| `debug = true` | `(True, True)` only |
| `dataset = "ESC50"` and `single = true` | `(False, False)` |
| `dataset = "ESC50"` and `single = false` | `(False, True)` and `(True, True)` |
| `dataset = "URBANSOUND8K"` and `single = false` | `(True, True)` and `(False, True)` |

Therefore, the number of submitted jobs is approximately:

```text
number of jobs = n_experiments × number of solution combinations
```

with the exception that `debug=true` forces `n_experiments=1` and only one solution.

If you want to run only one specific combination, edit `get_solutions()` in `experiment_utils/testing.py`.

---

## How the JSON fields map to the submitted command

For dataset `ESC50`, `testing.py` submits:

```bash
sbatch ESC50.sh <arguments...>
```

For dataset `URBANSOUND8K`, it submits:

```bash
sbatch URBANSOUND8K.sh <arguments...>
```

The important arguments are passed in this order:

| Position | Value passed by `testing.py` | Comes from |
|---:|---|---|
| 1 | `training_mode` | `cl_hyper.training_mode` |
| 2 | `cf_sol` | generated by `get_solutions()` |
| 3 | `head_sol` | generated by `get_solutions()` |
| 4 | `top_k` | `cl_hyper.top_k` |
| 5 | `high_lr` | `cl_hyper.high_lr` |
| 6 | `low_lr` | `cl_hyper.low_lr` |
| 7 | `t_criteria` | `cl_hyper.t_criteria` |
| 8 | `delta_w_interval` | `cl_hyper.delta_w_interval` |
| 9 | `selected_classes` | generated class split |
| 10 | `n_tasks` | derived from dataset/debug/single |
| 11 | `evaluated_tasks` | `evaluated_tasks` |
| 12 | `classes_per_task` | derived from dataset |
| 13 | `topk_lock` | `cl_hyper.topk_lock` |
| 14 | `folder_id` | generated from `id` and `n_tasks` |
| 15 | `parent_f_id` | generated from dataset and classes per task |
| 16 | `presets_path` | `presets_path` |
| 17 | `datasets_path` | `datasets_path` |
| 18 | `preset_name` | `preset_name` |
| 19 | `results_path` | `results_path` |

---

## Common configuration examples

### Debug run

Use this to check that paths, imports, dataset loading, and SLURM submission work.

```json
{
  "debug": true,
  "single": false,
  "n_experiments": 1,
  "dataset": "URBANSOUND8K",
  "id": "DEBUG_URBAN",
  "preset_name": "6SoftHebbCnnUrbanSound8k",
  "evaluated_tasks": [0, 1],
  "cl_hyper": {
    "training_mode": "consecutive",
    "top_k": 0.6,
    "topk_lock": false,
    "high_lr": 0.15,
    "low_lr": 0.9,
    "t_criteria": "activations",
    "delta_w_interval": 5
  }
}
```

Keep the path fields from your real machine/cluster configuration.

### Full ESC-50 incremental run

```json
{
  "debug": false,
  "single": false,
  "n_experiments": 2,
  "dataset": "ESC50",
  "id": "ESC50_FULL",
  "preset_name": "6SoftHebbCnnESC50",
  "datasets_path": "/scratch/project_462001198/casciott/datasets/ESC-50-master",
  "evaluated_tasks": [0, 1, 2, 3, 4],
  "cl_hyper": {
    "training_mode": "consecutive",
    "top_k": 0.6,
    "topk_lock": false,
    "high_lr": 0.15,
    "low_lr": 0.9,
    "t_criteria": "activations",
    "delta_w_interval": 5
  }
}
```

This submits two random class splits. Since normal ESC-50 runs submit `(False, True)` and `(True, True)`, this produces 4 jobs.

### Full UrbanSound8K incremental run

```json
{
  "debug": false,
  "single": false,
  "n_experiments": 3,
  "dataset": "URBANSOUND8K",
  "id": "URBAN_FULL",
  "preset_name": "6SoftHebbCnnUrbanSound8k",
  "datasets_path": "/scratch/project_462001198/casciott/datasets/urbansound8k",
  "evaluated_tasks": [0, 1, 2, 3, 4],
  "cl_hyper": {
    "training_mode": "consecutive",
    "top_k": 0.6,
    "topk_lock": false,
    "high_lr": 0.15,
    "low_lr": 0.9,
    "t_criteria": "activations",
    "delta_w_interval": 5
  }
}
```

This submits three random class splits. Since normal UrbanSound8K runs submit `(True, True)` and `(False, True)`, this produces 6 jobs.

---

## What the main continual-learning fields mean conceptually

### `cf_sol`

`cf_sol` enables the kernel-plasticity solution. When active, the code tracks Hebbian convolutional kernels, measures weight changes, ranks important kernels, and modifies learning-rate multipliers so that some kernels become more stable while others remain more plastic.

This field is generated by `testing.py`, not directly read from `experiment_config.json`.

### `head_sol`

`head_sol` enables the multi-head classifier solution. When active, the model keeps task-specific classifier heads and uses the corresponding head during evaluation.

This field is generated by `testing.py`, not directly read from `experiment_config.json`.

### `top_k`

`top_k` controls how many kernels are considered important. For example:

```json
"top_k": 0.6
```

means that about 60% of kernels are selected as top kernels according to the selected criterion.

### `high_lr` and `low_lr`

These values control relative learning-rate changes:

```text
non-protected kernels: multiplier roughly 1 + high_lr
protected kernels:     multiplier roughly 1 - low_lr
```

Examples:

```json
"high_lr": 0.15
```

increases selected plastic kernels by about 15%.

```json
"low_lr": 0.9
```

strongly reduces protected-kernel updates, giving a multiplier around 0.10 where the lower-learning-rate mask applies.

### `t_criteria`

The recommended value is:

```json
"t_criteria": "activations"
```

With `activations`, the model accumulates activation magnitude for each kernel and uses that ranking to select top kernels.

### `delta_w_interval`

This controls how often the code measures weight-change deltas during Hebbian training:

```json
"delta_w_interval": 5
```

means every 5 training iterations.

Smaller values update the measurements more often but add overhead. Larger values are cheaper but less fine-grained.

---

## Result JSON fields

Each experiment writes a JSON result file. Important fields include:

| Field | Meaning |
|---|---|
| `accuracy_matrix` | Main matrix for continual-learning performance. It stores accuracy after each training step on each evaluated task. |
| `performance_avg_folds` | Average task performance across folds. |
| `joint` | Joint-training baseline results over all classes seen so far. |
| `confusion_matrix` | Predictions and labels used to build confusion matrices, organized by fold/task. |
| `cl_hyper` | Continual-learning hyperparameters used for the run, including generated fields such as `selected_classes`. |
| `model_config` | Expanded model/block configuration loaded from `presets.json`. |
| `model_name` | Final model name, including the UUID appended by `continual_learning.py`. |
| `FOLD_#1`, `FOLD_#2`, ... | Per-fold training/evaluation summaries. ESC-50 uses 5 folds. UrbanSound8K uses 10 folds. |

---

## Common pitfalls

- **Dataset and preset mismatch**: if `dataset` is `URBANSOUND8K`, use `preset_name = "6SoftHebbCnnUrbanSound8k"`. If `dataset` is `ESC50`, use `preset_name = "6SoftHebbCnnESC50"`.
- **Wrong dataset root**: ESC-50 expects the root containing `meta/esc50.csv`; UrbanSound8K expects a root containing `h5s/urbansound8k.h5`.
- **Wrapper path mismatch**: `project_path` in the JSON does not automatically rewrite the hard-coded paths inside `experiment_utils/execs/*.sh`. Edit the wrappers manually if you move the repository.
- **Wrong evaluated-task argument**: the Python runner expects `--evaluated-tasks` with an `s`.
- **Reusing the same `id`**: if a result folder already exists, `testing.py` asks whether to continue. Use a new `id` for clean runs.
- **Expecting JSON to control `cf_sol`/`head_sol` directly**: these are currently chosen inside `get_solutions()` in `testing.py`.
- **Debug mode changes task count**: `debug=true` forces a reduced run and may not represent the full experiment structure.

---

## Baselines

The LwF/EWC baseline code is separate from `experiment_config.json` and is launched through:

```text
SoftHebb-main/baselines_LwF_EWC.py
experiment_utils/execs/baselines.sh
```

The JSON configuration described above is for the SoftHebb continual-learning experiments launched by `experiment_utils/testing.py`.
