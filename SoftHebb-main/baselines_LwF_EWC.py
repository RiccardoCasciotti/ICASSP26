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
# Dataset: ESC-50 H5 (mel-spectrogram) + constant task id
#   - returns LOCAL labels (0..C_task-1) for Task-IL multi-head training
# ----------------------------
class Esc50(Dataset):
    def __init__(
        self,
        data_path,
        selected_classes=list(range(50)),
        test=False,
        eval_fold=0,
        debug=False,
        task_idx: int = 0,
    ):
        self.h5_path = f"{data_path}/h5s/esc50.h5"
        self.dataset = h5py.File(self.h5_path, "r")

        self.selected_classes = sorted(selected_classes)

        # Map global class id -> local class id for this task
        self.class_to_local = {c: i for i, c in enumerate(self.selected_classes)}

        self.task_indexes = self.__get_task_dataset_indexes_from_hd5__(test, eval_fold, debug)
        np.random.shuffle(self.task_indexes)

        self.data = self.dataset["mel_spectrogram"]
        self.targets = self.dataset["labels_ids"]

        self.task_idx = int(task_idx)
        self.start = 0
        self.end = len(self.task_indexes)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        idx = self.task_indexes[self.start + index]
        y_global = int(self.targets[idx][0])
        y_local = self.class_to_local[y_global]
        x = torch.from_numpy(self.data[idx]).t().unsqueeze(0)  # [1, T, M]
        # Return (x, local_y, task_id)
        return x, y_local, self.task_idx

    def __get_task_dataset_indexes_from_hd5__(self, test=False, eval_fold=0, debug=False):
        data = []
        length = 500 if debug else len(self.dataset["filenames"])
        for i in range(length):
            entry = self.dataset["one_hot_labels"][i]
            res = np.any(entry[self.selected_classes])
            fold_check = self.dataset["folds"][i] == eval_fold
            if not test and res and not fold_check:
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

# Classification dataset helper compatibility
try:
    from avalanche.benchmarks.utils import make_classification_dataset  # type: ignore
    _HAVE_MAKE = True
except ImportError:
    from avalanche.benchmarks.utils import as_classification_dataset  # type: ignore
    _HAVE_MAKE = False

# Multi-task model base
try:
    from avalanche.models import MultiTaskModule  # type: ignore
except Exception:
    try:
        from avalanche.models.dynamic_modules import MultiTaskModule  # type: ignore
    except Exception as e:
        raise ImportError("Could not import MultiTaskModule from avalanche.") from e

from avalanche.training.plugins import SupervisedPlugin


# ----------------------------

class MultiHeadTiLNet(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int, n_classes_per_task: List[int]):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, n) for n in n_classes_per_task])
        self.current_task = 0

    def set_task(self, task_id: int):
        self.current_task = int(task_id)

    def forward_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        feats = self.backbone(x)
        return self.heads[int(task_id)](feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avalanche calls model(self.mb_x) with no task labels in your setup
        return self.forward_task(x, self.current_task)
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
    so LwF/EWC can run across avalanche versions.
    """
    if not hasattr(avl_ds, "targets_task_labels"):
        avl_ds.targets_task_labels = _TaskLabelsAttr(task_id, len(avl_ds))

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

    # Newer avalanche path: as_classification_dataset exists but can drop attrs.
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
    if not hasattr(avl, "targets"):
        avl.targets = list(targets)
    return ensure_task_attrs(avl, task_id)


# ----------------------------
# Helpers: folds, aligned targets (GLOBAL), stats, evaluation
# ----------------------------
def get_unique_folds(data_path: str) -> List[int]:
    # Keep your behavior (two folds)
    return [1, 2, 3, 4, 5]


def aligned_targets_global(ds: Esc50) -> np.ndarray:
    """
    Return GLOBAL class targets aligned with ds.__getitem__ order (ds.task_indexes order),
    while respecting h5py fancy-index requirement (increasing indices).
    """
    idxs = np.asarray(ds.task_indexes[ds.start : ds.end], dtype=np.int64)
    if idxs.size == 0:
        return np.asarray([], dtype=int)

    order = np.argsort(idxs)
    idxs_sorted = idxs[order]

    y_sorted = np.asarray(ds.targets[idxs_sorted])[:, 0].astype(int)

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return y_sorted[inv]


def to_local(ds: Esc50, y_global: np.ndarray) -> np.ndarray:
    if y_global.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray([ds.class_to_local[int(y)] for y in y_global], dtype=np.int64)


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
def eval_accuracy_til(model: MultiHeadTiLNet, ds: Dataset, device: str, batch_size: int):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    correct = total = 0

    for x, y, t in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # all samples in this loader belong to the same task, but we handle it generically:
        task_id = int(t[0].item())
        logits = model.forward_task(x, task_id)
        pred = logits.argmax(dim=1)
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


# ----------------------------
# Backbone CNN (unchanged from your version)
# ----------------------------
from torch.nn.parameter import UninitializedParameter


class BP6SoftHebbCnnImNet_BetterBP_ReLU(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_dim: int = 1000,
        dropout: float = 0.3,
        use_bn: bool = True,
        add_final_classifier: bool = True,
        conv_drop: float = 0.10,
        pool_k: int = 3,
        pool_s: int = 2,
        pool_p: int = 1,
    ):
        super().__init__()

        def BN(c):
            return nn.BatchNorm2d(c) if use_bn else nn.Identity()

        Act = nn.ReLU

        pool_max = nn.MaxPool2d(kernel_size=pool_k, stride=pool_s, padding=pool_p)

        # b0
        self.conv0 = nn.Conv2d(in_channels, 48, kernel_size=5, stride=1, padding=2, bias=not use_bn)
        self.bn0 = BN(48)
        self.act0 = Act(inplace=True)
        self.pool0 = pool_max

        # b1
        self.conv1 = nn.Conv2d(48, 192, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn1 = BN(192)
        self.act1 = Act(inplace=True)
        self.pool1 = pool_max

        # b2
        self.conv2 = nn.Conv2d(192, 768, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn2 = BN(768)
        self.act2 = Act(inplace=True)
        self.drop2 = nn.Dropout2d(p=conv_drop)
        self.pool2 = pool_max

        # b3
        self.conv3 = nn.Conv2d(768, 3072, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn3 = BN(3072)
        self.act3 = Act(inplace=True)
        self.drop3 = nn.Dropout2d(p=conv_drop)
        self.pool3 = pool_max

        # b4
        self.conv4 = nn.Conv2d(3072, 12288, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn4 = BN(12288)
        self.act4 = Act(inplace=True)
        self.drop4 = nn.Dropout2d(p=conv_drop)

        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # head
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(hidden_dim)

        self.add_final_classifier = bool(add_final_classifier)
        if self.add_final_classifier:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if isinstance(m.weight, UninitializedParameter):
                    continue
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.float()
        x = self.pool0(self.act0(self.bn0(self.conv0(x))))
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.drop2(self.act2(self.bn2(self.conv2(x)))))
        x = self.pool3(self.drop3(self.act3(self.bn3(self.conv3(x)))))
        x = self.pool4(self.drop4(self.act4(self.bn4(self.conv4(x)))))
        x = self.flatten(x)
        x = self.fc1(x)  # features [B, hidden_dim] if add_final_classifier=False
        x = self.head(x)
        return x


# ----------------------------
# Multi-head Task-IL model (one head per task)
# ----------------------------

# ----------------------------
# Plugin: freeze non-current heads per experience (recommended)
# ----------------------------

class FreezeNonCurrentHeadsPlugin(SupervisedPlugin):
    def __init__(self, model: MultiHeadTiLNet):
        super().__init__()
        self.m = model

    def _get_exp_task_id(self, exp) -> int:
        ds = getattr(exp, "dataset", None)
        if ds is not None and hasattr(ds, "targets_task_labels"):
            ttl = ds.targets_task_labels
            if hasattr(ttl, "uniques"):
                return int(ttl.uniques[0])
            try:
                return int(ttl[0])
            except Exception:
                pass
        return 0

    def before_training_exp(self, strategy, **kwargs):
        cur_t = self._get_exp_task_id(strategy.experience)
        self.m.set_task(cur_t)

        # backbone always trainable
        for p in self.m.backbone.parameters():
            p.requires_grad = True

        # only current head trainable
        for t, head in enumerate(self.m.heads):
            req = (t == cur_t)
            for p in head.parameters():
                p.requires_grad = req
# ----------------------------
# Build benchmark per fold (Task-IL: one experience per task)
# ----------------------------
def build_taskil_benchmark_for_fold(
    data_path: str,
    tasks: Sequence[Sequence[int]],
    eval_fold: int,
    debug: bool = False,
    num_classes_for_stats: int = 50,
):
    raw_train = []
    raw_test = []
    train_exps = []
    test_exps = []
    stats = []

    for task_idx, task_classes in enumerate(tasks):
        tr_ds = Esc50(
            data_path=data_path,
            selected_classes=list(task_classes),
            test=False,
            eval_fold=eval_fold,
            debug=debug,
            task_idx=task_idx,
        )
        te_ds = Esc50(
            data_path=data_path,
            selected_classes=list(task_classes),
            test=True,
            eval_fold=eval_fold,
            debug=debug,
            task_idx=task_idx,
        )

        # For stats: use GLOBAL labels (from H5) aligned to ds order
        y_tr_global = aligned_targets_global(tr_ds)
        y_te_global = aligned_targets_global(te_ds)

        stats.append(
            {
                "task_idx": task_idx,
                "classes": list(map(int, task_classes)),
                "train_n": int(len(tr_ds)),
                "test_n": int(len(te_ds)),
                "train_hist": class_histogram(y_tr_global, num_classes=num_classes_for_stats),
                "test_hist": class_histogram(y_te_global, num_classes=num_classes_for_stats),
            }
        )

        raw_train.append(tr_ds)
        raw_test.append(te_ds)

        # For Avalanche training/eval: use LOCAL labels
        y_tr_local = to_local(tr_ds, y_tr_global)
        y_te_local = to_local(te_ds, y_te_global)

        tr_av = wrap_as_avalanche_classification(tr_ds, targets=y_tr_local.tolist(), task_id=task_idx)
        te_av = wrap_as_avalanche_classification(te_ds, targets=y_te_local.tolist(), task_id=task_idx)

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
    stats_num_classes: int = 50,
    log_class_hists: bool = False,
) -> Tuple[float, List[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    bm, raw_train, raw_test, per_task_stats = build_taskil_benchmark_for_fold(
        data_path=data_path,
        tasks=tasks,
        eval_fold=eval_fold,
        debug=debug,
        num_classes_for_stats=stats_num_classes,
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

    hidden_dim = 1000

    # Backbone outputs FEATURES (hidden_dim) because add_final_classifier=False
    backbone = BP6SoftHebbCnnImNet_BetterBP_ReLU(
    in_channels=1,
    num_classes=1,
    hidden_dim=hidden_dim,
    add_final_classifier=False,
)

    model = MultiHeadTiLNet(
        backbone=backbone,
        hidden_dim=hidden_dim,
        n_classes_per_task=[len(t) for t in tasks],
    ).to(device)

    plugins = [FreezeNonCurrentHeadsPlugin(model)]

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
            eval_every=-1,
            plugins=plugins,
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
            plugins=plugins,
        )
    else:
        raise ValueError("strategy_name must be 'lwf' or 'ewc'")

    # Continual training (one experience per task)
    for exp in bm.train_stream:
        strategy.train(exp)

    # Manual per-task evaluation on each held-out task dataset (Task-IL, correct head)
    per_task_accs: List[float] = []
    total_correct = 0
    total_count = 0

    print(f"\n--- Fold {eval_fold} results ({strategy_name.upper()}, Task-IL multi-head) ---")
    for task_idx, te_ds in enumerate(raw_test):
        c, n, acc = eval_accuracy_til(model, te_ds, device=device, batch_size=eval_mb_size)
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
        default="/scratch/project_462001198/casciott/datasets/esc50",
        help="Root path containing h5s/esc50.h5",
    )
    ap.add_argument("--strategy", type=str, default="both", choices=["lwf", "ewc", "both"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug", action="store_true")

    # Stats histogram size (GLOBAL label space). 50 for ESC-50.
    ap.add_argument("--stats_num_classes", type=int, default=50)

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

############################################################################################################################################################
    ap.add_argument(
        "--tasks",
        type=str,
        default="[[34, 12, 11, 45, 31, 14, 28, 13, 25, 49, 46, 33, 15, 39, 29, 3, 47, 4, 44, 36, 35, 7, 23, 21, 1, 40, 0, 9, 41, 32], [5, 26, 27, 16, 38], [43, 10, 24, 20, 18], [37, 17, 22, 19, 30], [6, 42, 48, 2, 8]]",
        help="JSON list of tasks. Example: '[[0,1],[2,3],[4,5],[6,7],[8,9]]'",
    )

    args = ap.parse_args()
    tasks = parse_tasks(args.tasks)

    folds = get_unique_folds(args.data_path) ################################################################################################################
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
                stats_num_classes=args.stats_num_classes,
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