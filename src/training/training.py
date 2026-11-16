# SPDX-License-Identifier: MIT
# src/training/training.py

"""Module for training the model.

This module provides functions for loading train configurations and running
training.

Typical usage example:
    history = train_cd(model, device, train_loader, training_cfg, n_epochs, starting_epoch=0)

    or

    history = train_sm(model, device, train_loader, training_cfg, n_epochs, starting_epoch=0)
"""

import torch
from torch.profiler import record_function
import copy
import math
from collections.abc import Callable, Mapping, Sequence
from numbers import Real

from training.ptt import PTTCheckpoint


def _metric_value(value):
    """Convert scalar tensors to numbers and vector tensors to Python lists."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    return value


def _accumulate_metric(total, value):
    """Recursively accumulate numeric metric values without assuming metric names."""
    value = _metric_value(value)
    if value is None:
        return total

    if isinstance(value, Mapping):
        if total is None:
            total = {}
        if not isinstance(total, dict):
            raise TypeError("Metric structure changed between batches")
        for key, item in value.items():
            total[key] = _accumulate_metric(total.get(key), item)
        return total

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if total is None:
            total = [None] * len(value)
        if not isinstance(total, list) or len(total) != len(value):
            raise ValueError("Metric sequence shape changed between batches")
        return [
            _accumulate_metric(previous, item)
            for previous, item in zip(total, value)
        ]

    if not isinstance(value, Real):
        raise TypeError(f"Unsupported metric value type: {type(value).__name__}")
    if total is None:
        return (float(value), 1)
    if not (
        isinstance(total, tuple)
        and len(total) == 2
        and isinstance(total[1], int)
    ):
        raise TypeError("Metric structure changed between batches")
    return (total[0] + float(value), total[1] + 1)


def _average_metric(total):
    """Average an accumulated metric tree, preserving optional and nested values."""
    if isinstance(total, dict):
        return {key: _average_metric(value) for key, value in total.items()}
    if isinstance(total, list):
        return [_average_metric(value) for value in total]
    if total is None:
        return None
    if isinstance(total, tuple) and len(total) == 2:
        value, count = total
        return value / count if count else None
    raise TypeError(f"Invalid accumulated metric: {total!r}")


def _flatten_metrics(metrics: Mapping, prefix: str = "") -> dict:
    """Flatten nested metric dictionaries for the existing column-based history."""
    flattened = {}
    for key, value in metrics.items():
        key = str(key)
        if isinstance(value, Mapping):
            child_prefix = (
                f"{prefix}grad_norm_"
                if key == "grad_norms"
                else f"{prefix}{key}_"
            )
            flattened.update(_flatten_metrics(value, child_prefix))
        else:
            flattened[f"{prefix}{key}"] = value
    return flattened


def _append_epoch_metrics(history: dict, metrics: Mapping) -> None:
    """Append dynamically discovered metric columns, keeping every series aligned."""
    flattened = _flatten_metrics(metrics)
    epoch_count = max(
        (len(values) for key, values in history.items() if key != "pt_index_history"),
        default=0,
    )

    for key in history:
        if key != "pt_index_history" and key not in flattened:
            history[key].append(None)
    for key, value in flattened.items():
        if key not in history:
            history[key] = [None] * epoch_count
        history[key].append(value)


def finalize_epoch_metrics(epoch_metrics: dict, n_batches: int) -> dict:
    """Average the dynamically accumulated metrics for one epoch."""
    if n_batches == 0:
        raise ValueError("Cannot finalize metrics for an epoch with no batches")
    return _average_metric(epoch_metrics)


def overwrite_training_cfg(training_cfg: dict, schedule_node: dict) -> dict:
    """Return an overwritten version of training_cfg using an element of a schedule.

    Args:
        training_cfg: Training section/dictionary of the configuration dictionary parsed from the
            YAML configuration file.
        schedule_node: An element/node of a schedule list, containing values that can overwrite training_cfg.
    """
    training_cfg_overwritten = copy.deepcopy(training_cfg)

    for k in training_cfg_overwritten:
        if schedule_node.get(k) is not None:
            training_cfg_overwritten[k] = schedule_node.get(k)

    return training_cfg_overwritten


def _training_cfg_at_epoch(training_cfg: dict, epoch: int) -> dict:
    """Resolve the schedule values for an absolute epoch."""
    schedule = training_cfg.get("schedule", [{"start": 0}])
    schedule_idx = max(
        idx for idx, node in enumerate(schedule) if node["start"] <= epoch
    )
    schedule_node = schedule[schedule_idx]
    current = overwrite_training_cfg(training_cfg, schedule_node)

    transition_epochs = schedule_node.get("transition_epochs", 0)
    transition_type = schedule_node.get("transition_type", "linear")
    if schedule_idx == 0 or not transition_epochs:
        return current

    previous = overwrite_training_cfg(
        training_cfg, schedule[schedule_idx - 1]
    )
    progress = min(
        (epoch - schedule_node["start"] + 1) / transition_epochs, 1.0
    )
    for key in ("lr", "weight_decay", "momentum", "epsilon"):
        if schedule_node.get(key) is None:
            continue
        start_value = previous.get(key)
        end_value = current.get(key)
        if (
            isinstance(start_value, Real)
            and not isinstance(start_value, bool)
            and isinstance(end_value, Real)
            and not isinstance(end_value, bool)
        ):
            if transition_type == "linear":
                current[key] = start_value + (end_value - start_value) * progress
            elif transition_type == "exponential":
                current[key] = math.exp(
                    (1 - progress) * math.log(start_value)
                    + progress * math.log(end_value)
                )
            else:
                raise ValueError(
                    f"Unsupported schedule transition type: {transition_type}"
                )

    return current


def train_cd(
    model,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    training_cfg: dict,
    n_epochs: int,
    starting_epoch: int = 0,
    profiler=None,
    checkpoint_callback: Callable[[int, dict], None] | None = None,
) -> dict:
    """Train the RBM model using Contrastive Divergence or Persistent Contrastive Divergence.

    Args:
        model: RBM model instance.
        device: device outputted by get_device().
        train_loader: DataLoader for training data.
        training_cfg: contains parameters.
        n_epochs: Number of epochs to train in this session.
        starting_epoch: This is used to ensure that training is consistent with
            training.schedule even when training is stopped then resumed.
        profiler: Optional profiler for performance analysis.
        checkpoint_callback: Optional callback invoked with the absolute completed epoch and history.

    Returns:
        Dictionary with training metrics
    """
    schedule = training_cfg.get("schedule", [{"start": 0}])
    schedule_starts = {node["start"] for node in schedule}
    current = _training_cfg_at_epoch(training_cfg, starting_epoch)
    lr = current.get("lr")
    weight_decay = current.get("weight_decay")
    momentum = current.get("momentum")
    k = current.get("k")
    negative_phase_method = current.get("negative_phase_method")
    mc = current.get("mc")
    epsilon = current.get("epsilon")
    pt_n_chains = current.get("pt_n_chains")
    pt_max_T = current.get("pt_max_T")
    print(
        f"\nTraining with {negative_phase_method} and {k}-step {mc} updates. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}"
    )

    history = {"pt_index_history": []}

    for epoch in range(n_epochs):
        absolute_epoch = starting_epoch + epoch
        previous = (
            _training_cfg_at_epoch(training_cfg, absolute_epoch - 1)
            if absolute_epoch > 0
            else None
        )
        current = _training_cfg_at_epoch(training_cfg, absolute_epoch)
        lr = current.get("lr")
        weight_decay = current.get("weight_decay")
        momentum = current.get("momentum")
        k = current.get("k")
        negative_phase_method = current.get("negative_phase_method")
        mc = current.get("mc")
        epsilon = current.get("epsilon")
        pt_n_chains = current.get("pt_n_chains")
        pt_max_T = current.get("pt_max_T")

        if (
            previous is not None
            and previous.get("negative_phase_method")
            != negative_phase_method
        ):
            if (
                not negative_phase_method == "PCD"
                and model.persistent_v is not None
            ):
                model.persistent_v = None
                print(
                    "Deleted persistent batch, persistent contrastive divergence."
                )
            if (
                not negative_phase_method == "PT"
                and model.persistent_v_pt is not None
            ):
                model.persistent_v_pt = None
                model.pt_index_history = None
                model.pt_chain_perm = None
                print("Deleted persistent batch for parallel tempering.")

        if absolute_epoch in schedule_starts and absolute_epoch > starting_epoch:
            print(
                f"\nTraining with {negative_phase_method} and {k}-step {mc} updates. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}"
            )

        epoch_metrics_dict = {}
        for _, batch_data in enumerate(train_loader):
            X_train = (
                batch_data[0] if isinstance(batch_data, list) else batch_data
            )
            X_train = X_train.to(device)
            with record_function("train_batch"):
                batch_metrics_dict = model.train_batch(
                    X_train,
                    negative_phase_method,
                    mc,
                    k,
                    epsilon,
                    lr,
                    weight_decay,
                    momentum,
                    pt_n_chains,
                    pt_max_T,
                )
            if profiler is not None:
                profiler.step()

            if not isinstance(batch_metrics_dict, Mapping):
                raise TypeError("model.train_batch() must return a metric dictionary")
            for metric_name, metric_value in batch_metrics_dict.items():
                epoch_metrics_dict[metric_name] = _accumulate_metric(
                    epoch_metrics_dict.get(metric_name), metric_value
                )

        epoch_metrics = finalize_epoch_metrics(
            epoch_metrics_dict, len(train_loader)
        )
        _append_epoch_metrics(history, epoch_metrics)

        print(f"Epoch {epoch + 1}/{n_epochs}", end=", ")
        metrics = []
        for key, value in _flatten_metrics(epoch_metrics).items():
            if isinstance(value, Real):
                metrics.append(f"{key}: {value:.4f}")
            elif isinstance(value, list):
                metrics.append(f"{key}: list(len={len(value)})")
            else:
                metrics.append(f"{key}: {type(value).__name__}")
        print(", ".join(metrics))

        if checkpoint_callback is not None:
            checkpoint_callback(starting_epoch + epoch + 1, history)

    history["pt_index_history"].append(
        copy.deepcopy(model.pt_index_history)
        if model.pt_index_history is not None
        else None
    )

    return history


def train_sm(
    model,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    training_cfg: dict,
    n_epochs: int,
    starting_epoch: int = 0,
    profiler=None,
    checkpoint_callback: Callable[[int, dict], None] | None = None,
) -> dict:
    """Train the RBM model using Score-Matching.

    Parameters negative_phase_method, mc, k, and epsilon in training_cfg are only for diagnosis metrics calculation
    not for training itself.

    Notes:
        - If negative_phase_method is "PCD", pcd is used, else "CD" is used when calculating diagnostic metrics.

    Args:
        model: RBM model instance.
        device: device outputted by get_device().
        train_loader: DataLoader for training data.
        training_cfg: contains parameters.
        n_epochs: Number of epochs to train in this session.
        starting_epoch: This is used to ensure that training is consistent with
            training.schedule even when training is stopped then resumed.
        profiler: Optional profiler for performance analysis.
        checkpoint_callback: Optional callback invoked with the absolute completed epoch and history.

    Returns:
        Dictionary with training metrics
    """
    schedule = training_cfg.get("schedule", [{"start": 0}])
    schedule_starts = {node["start"] for node in schedule}
    current = _training_cfg_at_epoch(training_cfg, starting_epoch)
    lr = current.get("lr")
    weight_decay = current.get("weight_decay")
    momentum = current.get("momentum")
    k = current.get("k")
    negative_phase_method = current.get("negative_phase_method")
    mc = current.get("mc")
    epsilon = current.get("epsilon")
    print(
        f"\nTraining with score matching. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}"
    )

    history = {}

    optimizer = torch.optim.Adam(
        [
            {"params": [model.W], "lr": lr, "weight_decay": weight_decay},
            {"params": [model.z], "lr": lr * 0.1, "weight_decay": 0.0},
            {
                "params": [model.v_bias, model.h_bias],
                "lr": lr,
                "weight_decay": 0.0,
            },
        ]
    )

    for epoch in range(n_epochs):
        absolute_epoch = starting_epoch + epoch
        current = _training_cfg_at_epoch(training_cfg, absolute_epoch)
        lr = current.get("lr")
        weight_decay = current.get("weight_decay")
        momentum = current.get("momentum")
        k = current.get("k")
        negative_phase_method = current.get("negative_phase_method")
        mc = current.get("mc")
        epsilon = current.get("epsilon")
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[0]["weight_decay"] = weight_decay
        optimizer.param_groups[1]["lr"] = lr * 0.1
        optimizer.param_groups[2]["lr"] = lr

        if absolute_epoch in schedule_starts and absolute_epoch > starting_epoch:
            print(
                f"\nTraining with score matching. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}"
            )

        epoch_metrics_dict = {}
        for _, batch_data in enumerate(train_loader):
            X_train = (
                batch_data[0] if isinstance(batch_data, list) else batch_data
            )
            v = X_train.to(device)

            with record_function("train_sm_batch"):
                # Train
                optimizer.zero_grad()
                loss = model.score_matching_loss(v)
                loss.backward()
                optimizer.step()

                # Compute Energy and MSE for diagnosis
                with torch.no_grad():
                    # Initialize persistent chain the first time
                    if model.persistent_v is None:
                        model.persistent_v = v.detach().clone()

                    # Gibbs sampling
                    if negative_phase_method == "PCD":
                        model.persistent_v = model.persistent_v.detach()
                        v_sample = model.forward(
                            model.persistent_v, mc, k, epsilon
                        )  # [batch_size, nv]
                        model.persistent_v = v_sample.detach().clone()
                    else:  # CD
                        v_sample = model.forward(
                            v, mc, k, epsilon
                        )  # [batch_size, nv]

                    E_data = torch.mean(model.visible_energy(v))
                    E_model = torch.mean(model.visible_energy(v_sample))

                    E_diff = E_model - E_data

                    v_recon = model.forward(v, mc="gibbs", k=1)
                    mse = torch.mean(
                        (v_recon.clamp(0, 1) - v) ** 2
                    )  # clamp v' into [0,1]

            if profiler is not None:
                profiler.step()

            batch_metrics = {
                "E_data": E_data,
                "E_model": E_model,
                "E_diff": E_diff,
                "mse": mse,
                "loss": loss,
            }
            for metric_name, metric_value in batch_metrics.items():
                epoch_metrics_dict[metric_name] = _accumulate_metric(
                    epoch_metrics_dict.get(metric_name), metric_value
                )

        epoch_metrics = finalize_epoch_metrics(
            epoch_metrics_dict, len(train_loader)
        )
        _append_epoch_metrics(history, epoch_metrics)

        print(f"Epoch {epoch + 1}/{n_epochs}", end=", ")
        print(
            ", ".join(
                f"{key}: {value:.4f}"
                for key, value in _flatten_metrics(epoch_metrics).items()
            )
        )
        print(f"Average z: ", model.z.mean())

        if checkpoint_callback is not None:
            checkpoint_callback(starting_epoch + epoch + 1, history)
    return history


def train_ptt():
    raise NotImplementedError("PTT Not Implemented")

