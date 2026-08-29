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

from training.ptt import PTTCheckpoint


def zero_epoch_metrics() -> dict:
    """Return a dictionary of zero-valued training metrics for an epoch."""
    return {
        "E_data": 0.0,
        "E_model": 0.0,
        "E_diff": 0.0,
        "mse": 0.0,
        "loss": 0.0,
        "hidden_mean": 0.0,
        "grad_norm_W": 0.0,
        "grad_norm_v_bias": 0.0,
        "grad_norm_h_bias": 0.0,
        "pt_swap_acceptance": 0.0,
    }


def get_weight_grad_norm(diag_norms: dict) -> float:
    """Return a model-agnostic weight-gradient norm for generic metrics."""
    if "W" in diag_norms:
        return float(diag_norms["W"])
    if "A" in diag_norms and "B" in diag_norms:
        return float((diag_norms["A"] + diag_norms["B"]) / 2.0)
    return 0.0


def get_visible_bias_grad_norm(diag_norms: dict) -> float:
    """Return the visible-bias gradient norm if the model exposes one."""
    return float(diag_norms.get("v_bias", 0.0))


def finalize_epoch_metrics(
    epoch_metrics: dict, n_batches: int, pt_swap_count: int
) -> dict:
    """Average per-batch numbers for one epoch and finalize parallel-tempering stats."""
    for key in (
        "E_data",
        "E_model",
        "E_diff",
        "mse",
        "loss",
        "hidden_mean",
        "grad_norm_W",
        "grad_norm_v_bias",
        "grad_norm_h_bias",
    ):
        epoch_metrics[key] /= n_batches

    if pt_swap_count > 0:
        epoch_metrics["pt_swap_acceptance"] /= pt_swap_count
    else:
        epoch_metrics["pt_swap_acceptance"] = None

    return epoch_metrics


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


def train_cd(
    model,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    training_cfg: dict,
    n_epochs: int,
    starting_epoch: int = 0,
    profiler=None,
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

    Returns:
        Dictionary with training metrics
    """
    schedule = training_cfg.get("schedule", [{"start": 0}])
    epoch_to_idx = {schedule[i]["start"]: i for i in range(len(schedule))}

    starting_idx = 0
    for k in list(epoch_to_idx.keys()):
        if starting_epoch >= k:
            starting_idx = epoch_to_idx.pop(k)

    current = overwrite_training_cfg(training_cfg, schedule[starting_idx])
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

    history = {
        "E_data": [],
        "E_model": [],
        "E_diff": [],
        "mse": [],
        "loss": [],
        "hidden_mean": [],
        "grad_norm_W": [],
        "grad_norm_v_bias": [],
        "grad_norm_h_bias": [],
        "pt_swap_acceptance": [],
        "pt_index_history": [],
    }

    for epoch in range(n_epochs):
        if starting_epoch + epoch in epoch_to_idx:
            current = overwrite_training_cfg(
                training_cfg, schedule[epoch_to_idx[starting_epoch + epoch]]
            )
            lr = current.get("lr")
            weight_decay = current.get("weight_decay")
            momentum = current.get("momentum")
            k = current.get("k")
            negative_phase_method = current.get("negative_phase_method")
            mc = current.get("mc")
            epsilon = current.get("epsilon")

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

            print(
                f"\nTraining with {negative_phase_method} and {k}-step {mc} updates. lr={lr}, weight_decay={weight_decay}, momentum={momentum}, epsilon={epsilon}"
            )

        epoch_metrics = zero_epoch_metrics()
        pt_swap_count = 0
        for _, batch_data in enumerate(train_loader):
            X_train = (
                batch_data[0] if isinstance(batch_data, list) else batch_data
            )
            X_train = X_train.to(device)
            with record_function("train_batch"):
                E_data, E_model, E_diff, mse, loss, diagnostics = (
                    model.train_batch(
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
                )
            if profiler is not None:
                profiler.step()

            epoch_metrics["E_data"] += E_data.item()
            epoch_metrics["E_model"] += E_model.item()
            epoch_metrics["E_diff"] += E_diff.item()
            epoch_metrics["mse"] += mse.item()
            epoch_metrics["loss"] += loss.item()
            epoch_metrics["hidden_mean"] += diagnostics["hidden_mean"]
            epoch_metrics["grad_norm_W"] += get_weight_grad_norm(
                diagnostics["grad_norms"]
            )
            epoch_metrics["grad_norm_v_bias"] += get_visible_bias_grad_norm(
                diagnostics["grad_norms"]
            )
            epoch_metrics["grad_norm_h_bias"] += diagnostics["grad_norms"][
                "h_bias"
            ]
            if diagnostics["pt_swap_acceptance"] is not None:
                epoch_metrics["pt_swap_acceptance"] += diagnostics[
                    "pt_swap_acceptance"
                ]
                pt_swap_count += 1

        epoch_metrics = finalize_epoch_metrics(
            epoch_metrics, len(train_loader), pt_swap_count
        )

        history["E_data"].append(epoch_metrics["E_data"])
        history["E_model"].append(epoch_metrics["E_model"])
        history["E_diff"].append(epoch_metrics["E_diff"])
        history["hidden_mean"].append(epoch_metrics["hidden_mean"])
        history["grad_norm_W"].append(epoch_metrics["grad_norm_W"])
        history["grad_norm_v_bias"].append(epoch_metrics["grad_norm_v_bias"])
        history["grad_norm_h_bias"].append(epoch_metrics["grad_norm_h_bias"])
        history["pt_swap_acceptance"].append(
            epoch_metrics["pt_swap_acceptance"]
        )
        history["mse"].append(epoch_metrics["mse"])
        history["loss"].append(epoch_metrics["loss"])

        print(f"Epoch {epoch + 1}/{n_epochs}", end=", ")
        metrics = []
        for key, value in history.items():
            if key == "pt_index_history":
                continue

            last = value[-1]
            if isinstance(last, (int, float)):
                metrics.append(f"{key}: {last:.4f}")
            elif isinstance(last, list):
                metrics.append(f"{key}: list(len={len(last)})")
            elif isinstance(last, dict):
                metrics.append(f"{key}: dict(len={len(last)})")
            else:
                metrics.append(f"{key}: {type(last).__name__}")
        print(", ".join(metrics))

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

    Returns:
        Dictionary with training metrics
    """
    # Load parameters based on how many epochs have been trained by the model.
    # Create a start_epoch: schedule_index dictionary for schedule.
    schedule = training_cfg.get("schedule", [{"start": 0}])
    epoch_to_idx = {schedule[i]["start"]: i for i in range(len(schedule))}

    # Iterate through schedule to find the right starting parameters.
    starting_idx = 0
    for k in list(epoch_to_idx.keys()):
        if starting_epoch >= k:
            starting_idx = epoch_to_idx.pop(k)
            print(epoch_to_idx)

    current = overwrite_training_cfg(training_cfg, schedule[starting_idx])
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

    history = {"E_data": [], "E_model": [], "E_diff": [], "mse": [], "loss": []}

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
        if starting_epoch + epoch in epoch_to_idx:
            current = overwrite_training_cfg(
                training_cfg, schedule[epoch_to_idx[starting_epoch + epoch]]
            )
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

        E_data_epoch, E_model_epoch, E_diff_epoch, loss_epoch, mse_epoch = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
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

            E_data_epoch += E_data.item()
            E_model_epoch += E_model.item()
            E_diff_epoch += E_diff.item()
            mse_epoch += mse.item()
            loss_epoch += loss.item()

        E_data_epoch /= len(train_loader)
        E_model_epoch /= len(train_loader)
        E_diff_epoch /= len(train_loader)
        mse_epoch /= len(train_loader)
        loss_epoch /= len(train_loader)

        history["E_data"].append(E_data_epoch)
        history["E_model"].append(E_model_epoch)
        history["E_diff"].append(E_diff_epoch)

        history["mse"].append(mse_epoch)
        history["loss"].append(loss_epoch)

        print(f"Epoch {epoch + 1}/{n_epochs}", end=", ")
        print(
            ", ".join(
                f"{key}: {value[-1]:.4f}" for key, value in history.items()
            )
        )
        print(f"Average z: ", model.z.mean())
    return history


def train_ptt():
    raise NotImplementedError("PTT Not Implemented")

