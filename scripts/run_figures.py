import argparse
from os import path
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import json

from utils.device import get_device
from utils.checkpoint import load_checkpoint

from data.data_loader import load_data

from utils.multinomial import onehot_to_categories, categories_to_grayscale
import utils.physics as physics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RBM model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=path.join("outputs", "checkpoints", "default_run", "checkpoint.pt"),
        help="Path to checkpoint file"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Set n_samples to specify which sample to use. If k_gen set to None, select the most recent sample with n_samples."
    )

    parser.add_argument(
        "--k_gen",
        type=int,
        default=None,
        help="Set k_gen to specify which sample to use. If n_samples set to None, select the most recent sample with k_gen."
    )

    return parser.parse_args()

def wrap(array, minim=-np.pi, maxim=np.pi):
    """
    Wrap angles in array to the range [minim, maxim).
    Input:
    - array: Tensor of angles to wrap
    - minim: Minimum angle (default -π)
    - maxim: Maximum angle (default π)

    Returns:
    - Wrapped angles in the range [minim, maxim)
    """
    width = maxim - minim  # = 2π
    return (array - minim) % width + minim

def main():
    args = parse_args()

    # Load device: Either CPU or CUDA
    device = get_device()

    # Get checkpoint
    ckpt_path = args.checkpoint
    rbm, ckpt = load_checkpoint(ckpt_path, device)

    if ckpt is None:
        raise ValueError(f"The checkpoint file was not loaded from the checkpoint file properly.")

    # Get config file from checkpoint
    config = ckpt["config"]

    data_type = config["data"]["type"]
    data_dir = config["data"]["data_dir"]
    batch_size = config["data"]["batch_size"]
    split = config["data"]["split"]
    binarize = config["data"]["binarize"]
    q = config["data"]["q"]
    T = config["data"]["T"]
    L = config["data"]["L"]

    model_type = config["model"]["type"]
    n_class = config["model"]["n_class"]
    n_visible = config["model"]["n_visible"]
    n_hidden = config["model"]["n_hidden"]
    mf = config["model"]["mf"]

    out_dir = config["output"]["base_dir"]
    run_name = config["output"]["run_name"]

    # Load test data.
    _, test_loader = load_data(data_type, data_dir, split, q, T, L, batch_size, binarize, model_type)
    batch = next(iter(test_loader))
    if isinstance(batch, (list, tuple)):
        X_test = torch.cat([x for x, *_ in test_loader], dim=0)
    else:
        X_test = torch.cat([x for x in test_loader], dim=0)

    if model_type == "multinomial":
        X_test = X_test.to(device).float().view(-1, n_visible * n_class)
    else:
        X_test = X_test.to(device).float().view(-1, n_visible)

    # Get directories
    checkpoints_dir = path.join(out_dir, "checkpoints", run_name)
    figures_dir = path.join(out_dir, "figures", run_name)
    history_dir = path.join(out_dir, "history", run_name)
    samples_dir = path.join(out_dir, "samples", run_name)

    # Get samples using metadata. Import as torch tensor for consistency with X_test.
    meta_path = path.join(samples_dir, f"metadata.json")
    with open(meta_path) as f:
        all_meta = json.load(f)

    # Filter by args, fall back to most recent
    matches = [m for m in all_meta
               if (args.n_samples is None or m["n_samples"] == args.n_samples)
               and (args.k_gen is None or m["k_gen"]     == args.k_gen)]

    if not matches:
        raise FileNotFoundError(f"No samples found for n_samples={args.n_samples}, k_gen={args.k_gen}")

    meta = matches[-1]  # most recent match

    n_samples = meta["n_samples"]
    k_gen = meta["k_gen"]

    X_recon = np.load(path.join(samples_dir, meta["recon_file"]), allow_pickle=True)
    X_gen = np.load(path.join(samples_dir, meta["gen_file"]), allow_pickle=True)
    X_recon = torch.tensor(X_recon)
    X_gen = torch.tensor(X_gen)

    #  Get hidden unites corresponding to compare test vs recon. Needs to be done before any transformations for multinomial RBM
    with torch.no_grad():
        h_test = rbm.v_to_h(X_test.float().to(device)).detach() 
        h_recon = rbm.v_to_h(X_recon.float().to(device)).detach()

    # If model type is multinomial, extract categories from OneHot encoded data.
    # If image convert to grayscale in [0, 1].
    if model_type == "multinomial":
        X_test = onehot_to_categories(X_test, n_visible, n_class)
        X_recon = onehot_to_categories(X_recon, n_visible, n_class)
        X_gen = onehot_to_categories(X_gen, n_visible, n_class)

        if data_type in ["mnist", "cifar10", "stl10"]:
            X_test = categories_to_grayscale(X_test, n_class)
            X_recon = categories_to_grayscale(X_recon, n_class)
            X_gen = categories_to_grayscale(X_gen, n_class)

    # Wrap angles if model is VonMises
    if model_type == "vonmises":
        X_test = wrap(X_test, -np.pi, np.pi)
        X_recon = wrap(X_recon, -np.pi, np.pi)
        X_gen = wrap(X_gen, -np.pi, np.pi)

    '''
    Make plots: History
    '''
    # Get history from checkpoint and Plot curves
    history = ckpt["history"]
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

    axes[0].plot(history["E_data"],'rx-')
    axes[0].plot(history["E_model"],'bx-')

    axes[1].plot(history["E_diff"],'gx-')
    axes[2].plot(history["mse"],'kx-')
    axes[3].plot(history["loss"],'kx-')


    axes[0].set_title('visible energy: data/model')
    axes[1].set_title('visible energy diff')
    axes[2].set_title('reconstruction mse')
    axes[3].set_title('loss')
    fig.savefig(path.join(figures_dir, 'training_curve.png'), dpi=600, bbox_inches='tight')
    print(f"File, 'training_curve.png', saved to {figures_dir}")

    '''
    Make plots: Ising, Potts, XY
    '''
    if data_type in ["ising", "xy", "potts"]:
        if data_type == "ising":
            E_test = physics.ising_energy(2 * X_test.cpu() - 1)
            E_recon = physics.ising_energy(2 * X_recon.cpu() - 1)
            E_gen = physics.ising_energy(2 * X_gen.cpu() - 1)
            M_test = physics.ising_magnetization(2 * X_test.cpu() - 1)
            M_recon = physics.ising_magnetization(2 * X_recon.cpu() - 1)
            M_gen = physics.ising_magnetization(2 * X_gen.cpu() - 1)

        elif data_type == "xy":
            E_test = physics.xy_energy(X_test.cpu())
            E_recon = physics.xy_energy(X_recon.cpu())
            E_gen = physics.xy_energy(X_gen.cpu())
            M_test = physics.xy_magnetization(X_test.cpu())
            M_recon = physics.xy_magnetization(X_recon.cpu())
            M_gen = physics.xy_magnetization(X_gen.cpu())

        elif data_type == "potts":
            E_test = physics.potts_energy(X_test.cpu().long())
            E_recon = physics.potts_energy(X_recon.cpu().long())
            E_gen = physics.potts_energy(X_gen.cpu().long())
            M_test = physics.potts_magnetization(X_test.cpu().long(), n_class)
            M_recon = physics.potts_magnetization(X_recon.cpu().long(), n_class)
            M_gen = physics.potts_magnetization(X_gen.cpu().long(), n_class)
        
        C_test = physics.heat_capacity(E_test, float(T))
        C_recon = physics.heat_capacity(E_recon, float(T))
        C_gen = physics.heat_capacity(E_gen, float(T))
        Chi_test = physics.susceptibility(M_test, float(T))
        Chi_recon = physics.susceptibility(M_recon, float(T))
        Chi_gen = physics.susceptibility(M_gen, float(T))

        M_test_mean = M_test.abs().mean()
        M_recon_mean = M_recon.abs().mean()
        M_gen_mean = M_gen.abs().mean()
        E_test_mean = E_test.mean()
        E_recon_mean = E_recon.mean()
        E_gen_mean = E_gen.mean()

        if X_test.dim() == 1:
            X_test = X_test.unsqueeze(0)  # Add batch dimension if input is a single configuration
        N = X_test.flatten(start_dim=1).shape[1]

        # Example data
        sets = ["E/N", "C/N", "M/N", "Chi/N"]

        # 4 observables for Test, Recon, and Gen. In order from E/N, C/N, M/N, Chi/N
        values = np.array([
            [E_test_mean / N, E_recon_mean / N, E_gen_mean / N],  # E/N, 
            [C_test / N, C_recon / N, C_gen / N],  # C/N, 
            [M_test_mean / N, M_recon_mean / N, M_gen_mean / N, ],  # M/N, 
            [Chi_test / N, Chi_recon / N, Chi_gen / N]  # Chi/N
        ])

        # Labels for the 4 bars
        categories = ["Test", "Reconstructed", "Generated"]

        # X positions
        x = np.arange(len(sets))

        # Width of each bar
        bar_width = 0.2

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        for i in range(3):
            bars = ax.bar(x + i * bar_width, values[:, i], width=bar_width, label=categories[i])
            # Add values above bars
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xticks(x + 1.5 * bar_width, sets)
        ax.set_ylabel("Value")
        ax.set_title("Bar Plot Example")
        ax.legend()

        plt.tight_layout()
        plt.savefig(path.join(figures_dir, 'physical_properties.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'physical_properties.png', saved to {figures_dir}")

    '''
    Make plots: wind_dir dataset
    '''
    if data_type == 'wind_dir':
        # Plot histogram distributions of test, recon, and gen
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))

        axes.hist((X_test.cpu().flatten(), X_recon.cpu().flatten(), X_gen.cpu().flatten()), 
                  bins=50, color=((0.4, 0, 0, 0.6), (0, 0.4, 0, 0.6), (0, 0, 0.4, 0.6)),
                  label=("Test", "Recon", "Gen"))
        axes.legend()
        axes.set_xlim(-np.pi, np.pi)
        axes.set_title("Test vs Recon vs Gen Angles")

        fig.savefig(path.join(figures_dir, 'Test_recon_gen_angles.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'Test_recon_gen_angles.png', saved to {figures_dir}")
    
    '''
    Make plots: protein dataset
    '''
    if data_type == 'protein':
        # Plot histogram distributions of phi and psi angles
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        fig.suptitle("Phi and psi Distributions")

        ax_test = axes[0]
        ax_recon = axes[1]
        ax_gen = axes[2]

        ax_test.hist(X_test.cpu(),
                     bins=50, color=((0, 0, 0.4, 0.6), (0.4, 0, 0, 0.6)),
                     label=("phi", "psi"))
        ax_test.legend()
        ax_test.set_xlim(-np.pi, np.pi)
        ax_test.set_title("Test")

        ax_recon.hist(X_recon.cpu(),
                     bins=50, color=((0, 0, 0.4, 0.6), (0.4, 0, 0, 0.6)),
                     label=("phi", "psi"))
        ax_recon.legend()
        ax_recon.set_xlim(-np.pi, np.pi)
        ax_recon.set_title("Recon")

        ax_gen.hist(X_gen.cpu(),
                    bins=50, color=((0, 0, 0.4, 0.6), (0.4, 0, 0, 0.6)),
                    label=("phi", "psi"))
        ax_gen.legend()
        ax_gen.set_xlim(-np.pi, np.pi)
        ax_gen.set_title("Generated")

        fig.savefig(path.join(figures_dir, 'phi_psi_angles.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'phi_psi_angles.png', saved to {figures_dir}")

        # Plot Ramchadran of test, recon, and gen
        test_phi = X_test.cpu()[:, 0]
        test_psi = X_test.cpu()[:, 1]

        recon_phi = X_recon.cpu()[:, 0]
        recon_psi = X_recon.cpu()[:, 1]

        gen_phi = X_gen.cpu()[:, 0]
        gen_psi = X_gen.cpu()[:, 1]

        fig, axes = plt.subplots(1, 3, figsize = (12, 4))

        ax_test = axes[0]
        ax_recon = axes[1]
        ax_gen = axes[2]   

        ax_test.scatter(test_phi, test_psi, alpha=0.5, s=1)
        ax_test.set_xlim(-np.pi, np.pi)
        ax_test.set_ylim(-np.pi, np.pi)
        ax_test.set_xlabel("Phi (rad)")
        ax_test.set_ylabel("Psi (rad)")
        ax_test.set_title("Ramachandran Plot | Test")

        ax_recon.scatter(recon_phi, recon_psi, alpha=0.5, s=1)
        ax_recon.set_xlim(-np.pi, np.pi)
        ax_recon.set_ylim(-np.pi, np.pi)
        ax_recon.set_xlabel("Phi (rad)")
        ax_recon.set_ylabel("Psi (rad)")
        ax_recon.set_title("Ramachandran Plot | Reconstructed")

        ax_gen.scatter(gen_phi, gen_psi, alpha=0.5, s=1)
        ax_gen.set_xlim(-np.pi, np.pi)
        ax_gen.set_ylim(-np.pi, np.pi)
        ax_gen.set_xlabel("Phi (rad)")
        ax_gen.set_ylabel("Psi (rad)")
        ax_gen.set_title("Ramachandran Plot | Generated")
        plt.grid(True)
        fig.savefig(path.join(figures_dir, 'ramachandran.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'ramachandran.png', saved to {figures_dir}")

    '''
    Preparation for Image Datasets
    '''
    # Check if dataset is an image dataset. Set dimensions if image dataset. Also specify cmap.
    if data_type in ["mnist", "cifar10", "stl10", "ising", "xy", "potts"]:
        image = True
        # if dataset is an image dataset:
        # infer p from data if L is None
        if L is not None:
            p = L
            if p != int(np.sqrt(X_test.shape[1])):
                raise ValueError(
                    f"L={L} does not match the input data size of {X_test.shape[1]} visible units. "
                    f"Expected L={int(np.sqrt(X_test.shape[1]))}. Set L to null to infer automatically if not using data types 'ising', 'xy', or 'potts'."
                )
        else:
            p = int(np.sqrt(X_test.shape[1]))
            print(f"L not specified, inferred p={p} from n_visible={X_test.shape[1]}")
        
        if model_type == "multinomial" and data_type == "potts":
            cmap = 'tab10'
        elif model_type == "vonmises":
            cmap = 'hsv'
        elif model_type == "binary" and mf == False:
            cmap = 'binary'
        else:
            cmap = 'gray'
    else:
        image = False
        p = None

    '''
    Make plots: Image datasets - Reconstruction
    '''
    if image:
        # Plot Test vs Recon Images
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            im = axes[0, i].imshow(X_test[i].cpu().view(p, p), cmap=cmap)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            axes[1, i].imshow(X_recon[i].cpu().view(p, p), cmap=cmap)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

        axes[0, 0].set_ylabel("Original", fontsize=12)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
        plt.tight_layout()
        cbar = fig.colorbar(im, ax=axes, shrink=0.8)
        if cmap == 'tab10':
            cbar.set_ticks(range(q))
        
        fig.savefig(path.join(figures_dir, 'test_vs_recon.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'test_vs_recon.png', saved to {figures_dir}")

        # Plot Test vs Recon Histograms
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            axes[0, i].hist(X_test[i].cpu().view(-1))
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            axes[1, i].hist(X_recon[i].cpu().view(-1)) 
            axes[1, i].set_yticks([])

        axes[0, 0].set_ylabel("Original", fontsize=12)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
        plt.tight_layout()

        fig.savefig(path.join(figures_dir, 'test_vs_recon_hist.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'test_vs_recon_hist.png', saved to {figures_dir}")

        # Plot Test vs Recon Hidden Units Histograms
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            axes[0, i].hist(h_test[i].cpu().view(-1), range=[-0.5, 1.5]) 
            axes[0, i].set_xlim(0.0, 1.0)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            axes[1, i].hist(h_recon[i].cpu().view(-1), range=[-0.5, 1.5])
            axes[1, i].set_xlim(0.0, 1.0)
            axes[1, i].set_yticks([])

        axes[0, 0].set_ylabel("Original", fontsize=12)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
        plt.tight_layout()

        fig.savefig(path.join(figures_dir, 'test_vs_recon_h_hist.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'test_vs_recon_h_hist.png', saved to {figures_dir}")

    '''
    Make plots: Image datasets - Generation
    '''
    if image:
        image_count = max(0, min(n_samples, 512))  # Limit to 512 images
        image_count = 2 ** math.floor(math.log2(image_count))  # Round down to nearest power of 2
        # cols >= rows, cols = 2 * rows for powers of 2
        cols = int(2 ** math.ceil(math.log2(image_count) / 2 + 0.5))
        rows = image_count // cols

        # See generated images
        fig = plt.figure(figsize=(cols, rows)) 
        for i in range(cols * rows):  # grid
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(X_gen[i].cpu().view(p, p), cmap=cmap, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

        fig.savefig(path.join(figures_dir, 'gen_images.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'gen_images.png', saved to {figures_dir}")

        # Check histogram distribution
        fig = plt.figure(figsize=(cols, rows)) 
        for i in range(cols * rows):  # grid
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.hist(X_gen[i].cpu().view(-1)) 
            ax.set_xticks([])
            ax.set_yticks([])

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

        fig.savefig(path.join(figures_dir, 'gen_hist.png'), dpi=600, bbox_inches='tight')
        print(f"File, 'gen_hist.png', saved to {figures_dir}")

    '''
    Make plots: Weights
    '''
    # Get Weights and plot as filters
    # cols >= rows, cols = 2 * rows for powers of 2
    cols = int(2 ** math.ceil(math.log2(n_hidden) / 2 + 0.5))
    rows = n_hidden // cols
    
    if model_type == 'vonmises':
        Weight_A = rbm.A.detach().cpu()
        Weight_B = rbm.B.detach().cpu()

        # Check weight distribution
        plot_weight_hist(Weight_A, figures_dir, 'weight_A_hist.png')
        plot_weight_hist(Weight_B, figures_dir, 'weight_B_hist.png')

        # Plot the full weight matrix
        plot_weight(Weight_A, figures_dir, file_name='weight_A.png')
        plot_weight(Weight_B, figures_dir, file_name='weight_B.png')

        # Visualize all weight filters as images
        if image:
            plot_weight_as_images(Weight_A, rows, cols, (p, p), figures_dir, file_name='weight_A_images.png')
            plot_weight_as_images(Weight_B, rows, cols, (p, p), figures_dir, file_name='weight_B_images.png')

        # Plot histograms of weight matrix filters per hidden unit
        plot_weight_as_hists_per_h(Weight_A, rows, cols, figures_dir, 'weight_A_hists')
        plot_weight_as_hists_per_h(Weight_B, rows, cols, figures_dir, 'weight_B_hists')
    else:
        Weight = rbm.W.detach().cpu()

        # Check weight distribution
        plot_weight_hist(Weight, figures_dir, 'weight_hist.png')

        # Plot the full weight matrix
        plot_weight(Weight, figures_dir, 'weight.png')

        # Visualize all weight filters as images
        if image and not model_type == "multinomial":
            plot_weight_as_images(Weight, rows, cols, (p, p), figures_dir, 'weight_images.png')

        # Plot histograms of weight matrix filters per hidden unit
        plot_weight_as_hists_per_h(Weight, rows, cols, figures_dir, 'weight_hists')

    return

def plot_weight_hist(weight, dir, file_name='weight_hist'):
    # Check weight distribution
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    axes.hist(weight.flatten(), bins=100)
    axes.set_xlabel("W_ij value")
    axes.set_yticks([])
    axes.set_title("Weight Distribution")

    fig.savefig(path.join(dir, file_name), dpi=600, bbox_inches='tight')
    print(f"File, '{file_name}', saved to {dir}")

def plot_weight(weight, dir, file_name='weight.png', vmin=None, vmax=None):
    # Plot the full weight matrix
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    img = axes.imshow(weight, cmap="gray", vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(img, location='left')
    fig.savefig(path.join(dir, file_name), dpi=600, bbox_inches='tight')
    print(f"File, '{file_name}', saved to {dir}")

def plot_weight_as_images(weight, rows, cols, dim, dir, file_name='weight_images.png', vmin=None, vmax=None):
    # Visualize all weight filters as images
    fig = plt.figure(figsize=(cols, rows)) 
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        im = ax.imshow(weight[i].reshape(dim), cmap="gray", aspect='auto', vmin=vmin, vmax=vmax)
        ax.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    # cbar = fig.colorbar(im, orientation='horizontal')  #, fraction=0.05, pad=0.02)
    fig.savefig(path.join(dir, 'weight_images.png'), dpi=600, bbox_inches='tight')
    print(f"File, '{file_name}', saved to {dir}")

def plot_weight_as_hists_per_h(weight, rows, cols, dir, filename='weight_hists'):
    # Plot histograms of weight matrix filters per hidden unit
    fig = plt.figure(figsize=(cols, rows)) 
    for i in range(rows * cols): 
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(weight[i].view(-1), bins=50)
        ax.set_yticks([])
        
    plt.tight_layout()
    fig.savefig(path.join(dir, filename), dpi=600, bbox_inches='tight')
    print(f"File, '{filename}', saved to {dir}")

if __name__ == "__main__":
    main()