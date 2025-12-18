#!/usr/bin/env python3
import argparse
import gc
import json
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Dataset: UrbanSound8K H5 (mel-spectrogram) + constant task id
# ----------------------------
class Urbansound8k(Dataset):
    def __init__(
        self,
        data_path: str,
        selected_classes,
        test: bool = False,
        eval_fold: int = 0,
        debug: bool = False,
        task_idx: int = 0,
    ):
        self.dataset = h5py.File(f"{data_path}/h5s/urbansound8k.h5", "r")
        self.selected_classes = sorted(selected_classes)
        self.task_indexes = self.__get_task_dataset_indexes_from_hd5__(test, eval_fold, debug)

        np.random.shuffle(self.task_indexes)
        self.data = self.dataset["mel_spectrogram"]
        self.targets = self.dataset["labels_ids"]  # you read labels as targets[idx][0]

        self.start = 0
        self.end = len(self.task_indexes)

        self.task_idx = int(task_idx)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        idx = self.task_indexes[self.start + index]
        y = int(self.targets[idx][0])
        x = torch.from_numpy(self.data[idx]).t().unsqueeze(0)  # [1, T, M]
        # Return triple (x, y, task_id) to be safe with Avalanche
        return x, y, self.task_idx

    def __get_task_dataset_indexes_from_hd5__(self, test=False, eval_fold=0, debug=False):
        data = []
        length = 2000 if debug else len(self.dataset["filenames"])
        for i in range(length):
            entry = self.dataset["one_hot_labels"][i]
            res = np.any(entry[self.selected_classes])
            fold_check = self.dataset["folds"][i] == eval_fold
            if (not test) and res and (not fold_check):
                data.append(i)
            elif test and res and fold_check:
                data.append(i)
        return data

    def close(self):
        try:
            if hasattr(self, "dataset") and self.dataset:
                self.dataset.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


# ----------------------------
# Avalanche imports (with fallback)
# ----------------------------
try:
    from avalanche.benchmarks import benchmark_from_datasets
except Exception as e:
    raise ImportError(
        "Could not import Avalanche benchmark helpers. Check your avalanche-lib installation/version."
    ) from e

try:
    from avalanche.training import LwF, EWC
except Exception:
    from avalanche.training.supervised import LwF, EWC

# classification dataset helper compatibility
try:
    from avalanche.benchmarks.utils import make_classification_dataset  # type: ignore
    _HAVE_MAKE = True
except ImportError:
    from avalanche.benchmarks.utils import as_classification_dataset  # type: ignore
    _HAVE_MAKE = False

from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger

# ----------------------------
# Minimal task-label object expected by Avalanche LwF plugin:
# exp.dataset.targets_task_labels.uniques
# ----------------------------
class _TaskLabelsAttr:
    def __init__(self, task_id: int, n: int):
        self._task_id = int(task_id)
        self._n = int(n)

    @property
    def uniques(self):
        return [self._task_id]

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return self._task_id


def ensure_task_attrs(avl_ds, task_id: int):
    """
    Ensure the Avalanche dataset has:
      - targets_task_labels with `.uniques`
      - task_set mapping task_id -> dataset
    so LwF can run even if Avalanche drops those attrs.
    """
    if not hasattr(avl_ds, "targets_task_labels"):
        avl_ds.targets_task_labels = _TaskLabelsAttr(task_id, len(avl_ds))

    # LwF also indexes exp.dataset.task_set[task_id] in some versions
    if not hasattr(avl_ds, "task_set"):
        avl_ds.task_set = {int(task_id): avl_ds}

    return avl_ds


def wrap_as_avalanche_classification(ds: Dataset, targets: List[int], task_id: int):
    """
    Version-proof wrapper: returns an Avalanche ClassificationDataset with
    required attrs for LwF/EWC.
    """
    if _HAVE_MAKE:
        avl = make_classification_dataset(ds, targets=targets, task_labels=task_id)
        return ensure_task_attrs(avl, task_id)

    # Newer avalanche path: as_classification_dataset exists but drops attrs sometimes.
    class _AttrWrap(Dataset):
        def __init__(self, base_ds, targets, task_id):
            self.base_ds = base_ds
            self.targets = list(targets)
            self.targets_task_labels = [int(task_id)] * len(base_ds)

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            return self.base_ds[idx]

    wrapped = _AttrWrap(ds, targets, task_id)
    avl = as_classification_dataset(wrapped)
    # force required attributes
    if not hasattr(avl, "targets"):
        avl.targets = list(targets)
    return ensure_task_attrs(avl, task_id)


# ----------------------------
# Simple mel-spectrogram model: input [B,1,T,M]
# ----------------------------
class MelCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.float()
        z = self.net(x).flatten(1)
        return self.fc(z)


# ----------------------------
# Helpers: folds, aligned targets, stats, evaluation
# ----------------------------
def get_unique_folds(data_path: str) -> List[int]:
    h5_path = f"{data_path}/h5s/urbansound8k.h5"
    with h5py.File(h5_path, "r") as f:
        folds = np.array(f["folds"])
    return [1, 2]
    # return sorted(int(x) for x in np.unique(folds))


def aligned_targets(ds: Urbansound8k) -> np.ndarray:
    """
    Return class targets aligned with ds.__getitem__ order (ds.task_indexes order),
    while respecting h5py's requirement that fancy-index lists are increasing.
    labels_ids appear to be shape (N,1) => take [:,0].
    """
    idxs = np.asarray(ds.task_indexes[ds.start: ds.end], dtype=np.int64)
    if idxs.size == 0:
        return np.asarray([], dtype=int)

    order = np.argsort(idxs)
    idxs_sorted = idxs[order]

    y_sorted = np.asarray(ds.targets[idxs_sorted])[:, 0].astype(int)

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return y_sorted[inv]


def class_histogram(y: np.ndarray, num_classes: int) -> np.ndarray:
    hist = np.zeros(num_classes, dtype=np.int64)
    if y.size > 0:
        vals, cnts = np.unique(y, return_counts=True)
        for v, c in zip(vals, cnts):
            iv = int(v)
            if 0 <= iv < num_classes:
                hist[iv] += int(c)
    return hist


@torch.no_grad()
def eval_accuracy(model: nn.Module, ds: Dataset, device: str, batch_size: int) -> Tuple[int, int, float]:
    """
    Returns (correct, total, acc). Robust to datasets yielding (x,y) or (x,y,t).
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, _t = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    acc = float(correct / total) if total > 0 else float("nan")
    return int(correct), int(total), acc


def close_raw_datasets(raw_train, raw_test):
    for ds in list(raw_train) + list(raw_test):
        try:
            ds.close()
        except Exception:
            pass


import torch
import torch.nn as nn


class Triangle(nn.Module):
    """
    Simple differentiable triangle activation:
        y = clamp(a - |x|, min=0)
    """
    def __init__(self, a: float):
        super().__init__()
        self.a = float(a)

    def forward(self, x):
        return torch.clamp(self.a - torch.abs(x), min=0.0)


class BP6SoftHebbCnnImNet(nn.Module):
    """
    Backprop version mirroring your 6SoftHebbCnnImNet blocks:

    b0: Conv(out=48,   k=5,p=2,s=1)  + Triangle(0.7) + MaxPool2d(k=4,s=2,p=1)
    b1: Conv(out=192,  k=3,p=1,s=1)  + Triangle(0.7) + MaxPool2d(k=4,s=2,p=1)
    b2: Conv(out=768,  k=3,p=1,s=1)  + Triangle(0.7) + MaxPool2d(k=4,s=2,p=1)
    b3: Conv(out=3072, k=3,p=1,s=1)  + Triangle(1.4) + MaxPool2d(k=4,s=2,p=1)
    b4: Conv(out=12288,k=3,p=1,s=1)  + Triangle(1.0) + AvgPool2d(k=2,s=2,p=0)
    b5: Flatten + Dropout(0.3) + Linear(... -> 1000) + (optional Linear(1000 -> num_classes))

    Notes:
    - Your config has batch_norm=false, so no BN by default.
    - Uses LazyLinear because the flatten dim depends on your input (T,F).
    """
    def __init__(
        self,
        in_channels: int = 1,      # 1 for mel-spectrogram, 3 for RGB
        num_classes: int = 10,     # your dataset classes
        hidden_dim: int = 1000,    # BP-c1000
        dropout: float = 0.3,
        use_bn: bool = False,
        add_final_classifier: bool = True,
    ):
        super().__init__()

        BN = nn.BatchNorm2d if use_bn else (lambda c: nn.Identity())

        # b0
        self.conv0 = nn.Conv2d(in_channels, 48, kernel_size=5, stride=1, padding=2, bias=not use_bn)
        self.bn0 = BN(48)
        self.act0 = Triangle(0.7)
        self.pool0 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        # b1
        self.conv1 = nn.Conv2d(48, 192, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn1 = BN(192)
        self.act1 = Triangle(0.7)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        # b2
        self.conv2 = nn.Conv2d(192, 768, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn2 = BN(768)
        self.act2 = Triangle(0.7)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        # b3
        self.conv3 = nn.Conv2d(768, 3072, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn3 = BN(3072)
        self.act3 = Triangle(1.4)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        # b4
        self.conv4 = nn.Conv2d(3072, 12288, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn4 = BN(12288)
        self.act4 = Triangle(1.0)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # b5
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p=dropout)
        self.fc1 = nn.LazyLinear(hidden_dim)

        self.add_final_classifier = bool(add_final_classifier)
        self.fc_out = nn.Linear(hidden_dim, num_classes) if self.add_final_classifier else nn.Identity()



    def forward(self, x):
        x = x.float()

        x = self.pool0(self.act0(self.bn0(self.conv0(x))))
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = self.pool4(self.act4(self.bn4(self.conv4(x))))

        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc1(x)      # -> 1000 (hidden_dim)
        x = self.fc_out(x)   # -> num_classes (if enabled)
        return x
# ----------------------------
# Build benchmark per fold (Task-IL: one experience per task)
# ----------------------------
def build_taskil_benchmark_for_fold(
    data_path: str,
    tasks: Sequence[Sequence[int]],
    eval_fold: int,
    debug: bool = False,
    num_classes: int = 10,
):
    raw_train = []
    raw_test = []
    train_exps = []
    test_exps = []
    stats = []

    for task_idx, task_classes in enumerate(tasks):
        tr_ds = Urbansound8k(
            data_path=data_path,
            selected_classes=list(task_classes),
            test=False,
            eval_fold=eval_fold,
            debug=debug,
            task_idx=task_idx,
        )
        te_ds = Urbansound8k(
            data_path=data_path,
            selected_classes=list(task_classes),
            test=True,
            eval_fold=eval_fold,
            debug=debug,
            task_idx=task_idx,
        )

        y_tr = aligned_targets(tr_ds)
        y_te = aligned_targets(te_ds)

        stats.append(
            {
                "task_idx": task_idx,
                "classes": list(map(int, task_classes)),
                "train_n": int(len(tr_ds)),
                "test_n": int(len(te_ds)),
                "train_hist": class_histogram(y_tr, num_classes=num_classes),
                "test_hist": class_histogram(y_te, num_classes=num_classes),
            }
        )

        raw_train.append(tr_ds)
        raw_test.append(te_ds)

        tr_av = wrap_as_avalanche_classification(tr_ds, targets=y_tr.tolist(), task_id=task_idx)
        te_av = wrap_as_avalanche_classification(te_ds, targets=y_te.tolist(), task_id=task_idx)

        train_exps.append(tr_av)
        test_exps.append(te_av)

    bm = benchmark_from_datasets(train=train_exps, test=test_exps)
    return bm, raw_train, raw_test, stats


# ----------------------------
# Training + logging per fold/per task + return per-task accs
# ----------------------------
def run_one_fold(
    data_path: str,
    tasks: Sequence[Sequence[int]],
    eval_fold: int,
    strategy_name: str,
    device: str,
    train_mb_size: int,
    eval_mb_size: int,
    train_epochs: int,
    lr: float,
    lwf_alpha: float,
    lwf_temperature: float,
    ewc_lambda: float,
    ewc_mode: str,
    debug: bool,
    seed: int,
    num_classes: int = 10,
    log_class_hists: bool = False,
) -> Tuple[float, List[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, raw_train, raw_test, per_task_stats = build_taskil_benchmark_for_fold(
        data_path=data_path,
        tasks=tasks,
        eval_fold=eval_fold,
        debug=debug,
        num_classes=num_classes,
    )

    print(f"\n--- Fold {eval_fold} stats ---")
    for s in per_task_stats:
        print(
            f"Task {s['task_idx']:02d} classes={s['classes']} | "
            f"train_n={s['train_n']} test_n={s['test_n']}"
        )
        if log_class_hists:
            print(f"  train_hist={s['train_hist'].tolist()}")
            print(f"  test_hist ={s['test_hist'].tolist()}")

    model = BP6SoftHebbCnnImNet(in_channels=1, num_classes=10, hidden_dim=1000, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()



    if strategy_name.lower() == "lwf":
        strategy = LwF(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        alpha=lwf_alpha,
        temperature=lwf_temperature,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=eval_mb_size,
        device=device,
        eval_every=-1,   # no eval during training => much less output
    )
    elif strategy_name.lower() == "ewc":

        strategy = EWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        ewc_lambda=ewc_lambda,
        mode=ewc_mode,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=eval_mb_size,
        device=device,
        eval_every=-1,
    )
    else:
        raise ValueError("strategy_name must be 'lwf' or 'ewc'")

    # Continual training (one experience per task)
    for exp in bm.train_stream:
        strategy.train(exp)

    # Manual per-task evaluation on each held-out task dataset (exact micro-average)
    per_task_accs: List[float] = []
    total_correct = 0
    total_count = 0

    print(f"\n--- Fold {eval_fold} results ({strategy_name.upper()}) ---")
    for task_idx, te_ds in enumerate(raw_test):
        c, n, acc = eval_accuracy(model, te_ds, device=device, batch_size=eval_mb_size)
        per_task_accs.append(acc)
        print(f"Task {task_idx:02d} test acc: {acc:.4f}  (n={n})")

        if n > 0:
            total_correct += c
            total_count += n

    overall = float(total_correct / total_count) if total_count > 0 else float("nan")
    print(f"Overall (micro avg over all test samples) fold acc: {overall:.4f}")

    close_raw_datasets(raw_train, raw_test)
    del bm, strategy, model
    gc.collect()

    return overall, per_task_accs


def parse_tasks(tasks_json: str) -> List[List[int]]:
    tasks = json.loads(tasks_json)
    if not isinstance(tasks, list) or not all(isinstance(t, list) for t in tasks):
        raise ValueError("tasks_json must be a JSON list of lists, e.g. [[0,1],[2,3]]")
    return [list(map(int, t)) for t in tasks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_path",
        type=str,
        default="/scratch/project_462001198/casciott/datasets/urbansound8k",
        help="Root path containing h5s/urbansound8k.h5",
    )
    ap.add_argument("--strategy", type=str, default="both", choices=["lwf", "ewc", "both"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--num_classes", type=int, default=10)

    ap.add_argument("--train_mb_size", type=int, default=64)
    ap.add_argument("--eval_mb_size", type=int, default=128)
    ap.add_argument("--train_epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--lwf_alpha", type=float, default=1.0)
    ap.add_argument("--lwf_temperature", type=float, default=2.0)

    ap.add_argument("--ewc_lambda", type=float, default=5.0)
    ap.add_argument(
        "--ewc_mode",
        type=str,
        default="separate",
        choices=["separate", "onlinesum", "onlineweightedsum"],
    )

    ap.add_argument("--log_class_hists", action="store_true")

    ap.add_argument(
        "--tasks",
        type=str,
        default="[[0,1],[2,3],[4,5],[6,7],[8,9]]",
        help="JSON list of tasks. Example: '[[0,1],[2,3],[4,5],[6,7],[8,9]]'",
    )

    args = ap.parse_args()
    tasks = parse_tasks(args.tasks)

    folds = get_unique_folds(args.data_path)
    if len(folds) < 2:
        raise RuntimeError(f"Found folds={folds}. Expected multiple folds in H5 'folds' dataset.")

    strategies = ["lwf", "ewc"] if args.strategy == "both" else [args.strategy]

    for strat in strategies:
        print(f"\n========== Strategy: {strat.upper()} ==========")

        fold_overalls: List[float] = []
        task_accs_across_folds: Dict[int, List[float]] = {i: [] for i in range(len(tasks))}

        for f in folds:
            overall, per_task_accs = run_one_fold(
                data_path=args.data_path,
                tasks=tasks,
                eval_fold=f,
                strategy_name=strat,
                device=args.device,
                train_mb_size=args.train_mb_size,
                eval_mb_size=args.eval_mb_size,
                train_epochs=args.train_epochs,
                lr=args.lr,
                lwf_alpha=args.lwf_alpha,
                lwf_temperature=args.lwf_temperature,
                ewc_lambda=args.ewc_lambda,
                ewc_mode=args.ewc_mode,
                debug=args.debug,
                seed=args.seed,
                num_classes=args.num_classes,
                log_class_hists=args.log_class_hists,
            )

            fold_overalls.append(overall)
            for task_idx, acc in enumerate(per_task_accs):
                task_accs_across_folds[task_idx].append(acc)

            print(f"\n[{strat.upper()}] held-out fold {f}: overall acc={overall:.4f}")

        mean_overall = float(np.nanmean(fold_overalls))
        std_overall = float(np.nanstd(fold_overalls))
        print(f"\n[{strat.upper()}] Overall accuracy across folds: {mean_overall:.4f} ± {std_overall:.4f}")

        print(f"\n[{strat.upper()}] Per-task accuracy averaged across folds:")
        for task_idx in range(len(tasks)):
            vals = np.array(task_accs_across_folds[task_idx], dtype=np.float32)
            m = float(np.nanmean(vals))
            s = float(np.nanstd(vals))
            print(f"  Task {task_idx:02d} classes={tasks[task_idx]}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()