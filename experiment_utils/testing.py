import json
import os
import random
import shlex
import subprocess
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def get_dataset_settings(dataset: str, single: bool):
    dataset = dataset.upper()

    if dataset == "ESC50":
        classes_per_task = 5

        if single:
            n_tasks = 1
        else:
            n_tasks = 5

        all_classes = list(range(50))

    elif dataset == "URBANSOUND8K":
        classes_per_task = 2

        if single:
            n_tasks = 1
        else:
            n_tasks = 5

        all_classes = list(range(10))

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return n_tasks, classes_per_task, all_classes


def build_folder_ids(config: dict, dataset: str, n_tasks: int, classes_per_task: int):
    exp_id = config["id"]

    folder_id = f"_{exp_id}{n_tasks}tasks"
    parent_f_id = f"experiments/EXP_{dataset}_{classes_per_task}C"

    return folder_id, parent_f_id


def ensure_experiment_folders(base_path: str, parent_f_id: str) -> None:
    base_path = Path(base_path)

    experiments_dir = base_path / "experiments"
    parent_dir = base_path / parent_f_id

    experiments_dir.mkdir(parents=True, exist_ok=True)
    parent_dir.mkdir(parents=True, exist_ok=True)


def folder_exists(base_path: str, relative_path: str) -> bool:
    path = Path(base_path) / relative_path
    print(f"Checking folder: {path}")
    return path.is_dir()


def generate_class_splits(
    dataset: str,
    n_experiments: int,
    n_tasks: int,
    classes_per_task: int,
    all_classes: list,
    single: bool,
    test: bool
):

    if test:
        n_experiments = 1
        n_tasks = 2

    if single:
        return [[all_classes.copy()]]

    classes = []
    dataset = dataset.upper()

    if dataset == "ESC50":
        for _ in range(n_experiments):
            shuffled_classes = all_classes.copy()
            random.shuffle(shuffled_classes)

            task_classes = []
            task_classes.append(shuffled_classes[:30])

            for i in range(30, 50, 5):
                task_classes.append(shuffled_classes[i:i + 5])

            classes.append(task_classes)

    elif dataset == "URBANSOUND8K":
        for _ in range(n_experiments):
            shuffled_classes = all_classes.copy()
            random.shuffle(shuffled_classes)

            task_classes = []
            task_classes.append(shuffled_classes[:classes_per_task])

            for i in range(classes_per_task, n_tasks * classes_per_task, classes_per_task):
                task_classes.append(shuffled_classes[i:i + classes_per_task])

            classes.append(task_classes)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return classes


def get_solutions(dataset: str, single: bool, test: bool):
    if test:
        return [(True, True)]

    if dataset.upper() == "ESC50":
        if single:
            return [(False, False)]
        else:
            return [(False, True), (True, True)]

    return [(True, True), (False, True)]


def build_shell_command(
    config: dict,
    cl_hyper: dict,
    selected_classes: list,
    evaluated_tasks: list,
    n_tasks: int,
    classes_per_task: int,
    folder_id: str,
    parent_f_id: str,
    datasets_path: str,
    presets_path: str,
    preset_name: str, 
    results_path: str
):
    """
    Runs the original .sh file directly.

    Equivalent to the original command structure:

    cd /.../experiment_utils/execs/ && sbatch ESC50.sh <args>

    but without creating configs.sh.
    """

    dataset = config["dataset"].upper()
    execs_dir = config["execs_dir"]
    sh_file = f"{dataset}.sh"

    selected_classes_str = json.dumps(selected_classes)
    evaluated_tasks_str = json.dumps(evaluated_tasks)

    command = [
        "sbatch",
        sh_file,

        str(cl_hyper["training_mode"]),
        str(cl_hyper["cf_sol"]),
        str(cl_hyper["head_sol"]),
        str(cl_hyper["top_k"]),
        str(cl_hyper["high_lr"]),
        str(cl_hyper["low_lr"]),
        str(cl_hyper["t_criteria"]),
        str(cl_hyper["delta_w_interval"]),

        f'{selected_classes_str}',
        str(n_tasks),
        f'{evaluated_tasks_str}',
        str(classes_per_task),
        str(cl_hyper["topk_lock"]),
        str(folder_id),
        str(parent_f_id),
        str(presets_path),
        str(datasets_path), 
        str(preset_name),
        str(results_path)

    ]

    return command


def run_experiments(config_path: str):
    config = load_config(config_path)

    test = config["debug"]
    single = config["single"]

    dataset = config["dataset"].upper()
    n_experiments = config["n_experiments"]

    if test:
        n_experiments = 1

    n_tasks, classes_per_task, all_classes = get_dataset_settings(dataset, single)

    if test:
        n_tasks = 2

    folder_id, parent_f_id = build_folder_ids(
        config=config,
        dataset=dataset,
        n_tasks=n_tasks,
        classes_per_task=classes_per_task
    )

    base_path = config["base_path"]
    presets_path = config["presets_path"]
    datasets_path = config["datasets_path"]
    preset_name=config["preset_name"]
    results_path=config["results_path"]
    
    ensure_experiment_folders(base_path, parent_f_id)

    cl_hyper = config["cl_hyper"].copy()
    cl_hyper["n_tasks"] = n_tasks
    cl_hyper["classes_per_task"] = classes_per_task
    cl_hyper["SINGLE"] = single

    evaluated_tasks = config["evaluated_tasks"]

    class_splits = generate_class_splits(
        dataset=dataset,
        n_experiments=n_experiments,
        n_tasks=n_tasks,
        classes_per_task=classes_per_task,
        all_classes=all_classes,
        single=single,
        test=test
    )

    print(f"Number of experiments: {n_experiments}")
    print("Classes selected for each task in each experiment:")
    print(json.dumps(class_splits, indent=2))

    target_folder = f"{parent_f_id}/TASKS_CL_{dataset + folder_id}"

    if folder_exists(base_path, target_folder):
        response = input(
            f"!!!! WARNING: A folder named 'TASKS_CL_{dataset + folder_id}' already exists. "
            f"Press Y to continue anyway or N to abort: "
        )

        if response != "Y":
            print("Aborted.")
            return

    solutions = get_solutions(dataset, single, test)

    for cf_sol, head_sol in solutions:
        cl_hyper["cf_sol"] = cf_sol
        cl_hyper["head_sol"] = head_sol
        cl_hyper["classes_per_task"] = classes_per_task

        for selected_classes in class_splits:
            command = build_shell_command(
                config=config,
                cl_hyper=cl_hyper,
                selected_classes=selected_classes,
                evaluated_tasks=evaluated_tasks,
                n_tasks=n_tasks,
                classes_per_task=classes_per_task,
                folder_id=folder_id,
                parent_f_id=parent_f_id,
                datasets_path=datasets_path,
                presets_path=presets_path,
                preset_name=preset_name,
                results_path=results_path
            )

            print("\nSubmitting command:")
            print(" ".join(shlex.quote(arg) for arg in command))

            subprocess.run(
                command,
                cwd=config["execs_dir"],
                check=True
            )

        if test:
            print("!!!! WARNING: BREAK OPERATION IS ON IN DEBUGGING")
            break


if __name__ == "__main__":
    path = input("Insert the full path for the experiment configuration json file: ")
    run_experiments(path)