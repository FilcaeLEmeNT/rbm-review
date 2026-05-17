# RBM in Physics

This repository accompanies the review paper

**“Restricted Boltzmann Machines in Physics: Concepts, Theories, and Applications”**  
by Kai Zhang and Sora Sakai.

It provides sample Python code and model implementations mentioned in the paper.

---

## 📁 Repo Structure

```
rbm-review/
├── configs
│   ├── default.yaml
├── data
├── LICENSE
├── notebooks
│   ├── RBM_binary.ipynb
│   ├── RBM_exponential.ipynb
│   ├── RBM_gaussian.ipynb
│   ├── RBM_multinomial.ipynb
│   └── RBM_vonmises.ipynb
├── outputs
│   ├── checkpoints
│   ├── figures
│   ├── history
│   └── samples
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
│   ├── run_train.py
|   ├── run_sample.py
|   └── run_figures.py
└── src
    ├── data
    │   ├── data_loader.py
    ├── models
    │   ├── rbm_binary.py
    │   ├── rbm_exponential.py
    │   ├── rbm_gaussian.py
    │   ├── rbm_multinomial.py
    │   └── rbm_vonmises.py
    ├── training
    │   └── training.py
    └── utils
        ├── config.py
        ├── device.py
        ├── checkpoint.py
        ├── physics.py
```

### Description
- `configs/` – YAML configuration files for training hyperparameters and setup
- `data/` – Dataset directory for data storage (raw and processed)
- `notebooks/` – Jupyter notebooks demonstrating each RBM architecture.
- `outputs/` – Stores training outputs, including checkpoints, figures, and model samples.
- `scripts/` – Excecutrion scripts for training and evaluation.
- `src/data/` – Data loading utilities.
- `src/models/` – Implementations of different RBM architectures.
- `src/training/` – Training loops and optimization code.
- `src/utils/` – Utility functions for config loading, device management, checkpoint saving/loading, and physical observables.

### Jupyter Notebooks and Scripts

Jupyter notebooks contained in `notebooks/` are standalone and can be run independently. These are primarily used for visualization.

However, this repository also includes scripts that can be run in the command prompt.

---

## ⚙️ Setup

### 1. Create virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate    # Windows
```

### 2. Install dependencies.
```bash
pip install -r requirements.txt
pip install -e .
```

## 🔧 Configuration

The configs directory contains YAML files that can are used as an argument for specifying parameters and hyperparameters for training, used like:
```bash
python scripts/run_train.py --config configs/default.yaml
``` 
Detailed descriptions of each parameter is highlighted in the default.yaml configuration file in the configs directory.

## 🗂️ Datasets

In the config file, data.type can be set to 'mnist', 'stl10', 'cifar10', 'ising', 'xy', 'potts', 'wind_dir', or 'protein'.
- **For 'mnist', 'stl10', and 'cifar10'**: The dataset will be automatically downloaded to the directory specified by data.data_dir in the config file.
- **For 'ising', 'xy', 'potts'**: Download the dataset from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19712829.svg)](https://doi.org/10.5281/zenodo.19712829) and put it inside the directory specified by data.data_dir in the config file.
- **For 'wind_dir'**: Download the dataset from [this website](https://www.ndbc.noaa.gov/download_data.php?filename=42503e2023.txt.gz&dir=data/historical/drift/) and put it inside the directory specified by data.data_dir in the config file. The point of this is to provide an example of learning a dataset with one angular visible unit, wind direction.
- **For 'protein'**: A Sidechain Net ([Link to Github](https://github.com/jonathanking/sidechainnet)) dataset will be automatically downloaded to the directory specified by data.data_dir in the config file. Moreover, two dihedral angles, phi and psi are automatically parsed to create a dataset that is loaded while training and sampling. The dataset may take a while to download.

## 🚗 Running Training

Example prompt:

```bash
python scripts/run_train.py --config configs/default.yaml  # Linux / Mac

# python scripts\run_train.py --config configs\default.yaml  # Windows
``` 
or
```bash
python -m scripts.run_train --config configs/default.yaml   # Linux / Mac

# python -m scripts.run_train --config configs\default.yaml # Windows
``` 
### Arguments
- '--config': Path to the configuration file that specifies the dataset, model architecture, training hyperparameters, etc.

## 🚩 Checkpoints

Upon running training, a checkpoint is saved to `outputs/checkpoints/default_run/`, assuming output.base_dir is set to 'outputs' and run_name is set to 'default_run' in the configs file. It contains everything needed to reconstruct the model without any additional arguments:

```python
{
    "epoch": int,
    "model_state": dict,  # model weights — e.g. W, b_v, b_h
    "optimizer_state": dict,
    "config": dict,  # architecture + hyperparameters
    "history": dict,  # train loss/energy history
}
```

Because the config is embedded, downstream scripts take `--checkpoint` as their only required argument. No separate `--config` needed:

```bash
python scripts/run_eval.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt

python scripts/run_sample.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt

python scripts/run_figures.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt
```

To load a checkpoint:

```python
from utils.checkpoint import load_checkpoint

model, ckpt = load_checkpoint(ckpt_path, device)
```

## 🏭 Sampling

Example prompt:

```bash
python scripts/run_sample.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt --k_gen 1000 --n_samples 8192  # Linux / Mac

# python scripts\run_sample.py --checkpoint outputs\checkpoints\default_run\checkpoint.pt --k_gen 1000 --n_samples 8192  # Windows
``` 

### Arguments
- '--checkpoint': Path to the checkpoint file generated by run_train.py
- '--k_gen': Amount of k steps used to generate samples from a randomized tensor. Defaults to 1000.
- '--n_samples': Amount of samples generated from the script. Defaults to 8192.

### Outputs
These files are saved to `outputs/samples/default_run/` (the path depends on the output settings in the config file)
- `recon.npy`: Reconstructions of the test data after one Gibbs/Langevin step from each test sample. Shape `(N_test, n_visible)`. Used to assess reconstruction quality.
- `gen_n{n_samples}_k{k_gen}.npy`: Freely generated samples produced by running `k_gen` Gibbs/Langevin steps from a random initial state. Shape `(n_samples, n_visible)`. Used for plotting in `run_figures.py`.
- `metadata.json`: A log of all sampling runs performed on this checkpoint. Each entry records the checkpoint path, `n_samples`, `k_gen`, the corresponding output filenames, and a timestamp — so any figure can be traced back to the exact sampling run that produced it.

### Notes
- `recon.npy` starts from real test data and takes one step. It stays close to the test data.
- `gen_n{n_samples}_k{k_gen}.npy` starts from random noise and runs a long chain. These are the true model samples used for evaluating the learned distribution.
- If the model was trained with Gibbs updates, samples will be generated using Gibbs sampling. If trained with Langevin updates, Langevin steps will be used instead.

## 📊 Figures
Example script:
```bash
python scripts/run_figures.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt --k_gen 1000 --n_samples 8192  # Linux / Mac

# python scripts\run_figures.py --checkpoint outputs\checkpoints\default_run\checkpoint.pt --k_gen 1000 --n_samples 8192  # Windows
```

### Arguments
- `--checkpoint`: Path to the checkpoint file generated by `run_train.py`
- `--k_gen`: Selects which saved sample file to use by matching `k_gen`. If omitted, the most recent sample matching `n_samples` is used.
- `--n_samples`: Selects which saved sample file to use by matching `n_samples`. If omitted, the most recent sample matching `k_gen` is used.

### Outputs
These files are saved to `outputs/figures/default_run/` (the path depends on the output settings in the config file)

**All models:**
- `training_curve.png`: Energy of data and model configurations, energy difference, reconstruction MSE, and reconstruction cross-entropy over training epochs.
- `weight_hist.png`: Distribution of weight matrix values.
- `weight.png`: Full weight matrix as a heatmap.
- `weight_hists.png`: Per-hidden-unit weight distributions.

**Von Mises RBM (has two weight matrices A and B instead of W):**
- `weight_A_hist.png`, `weight_B_hist.png`: Distributions of the A and B weight matrices.
- `weight_A.png`, `weight_B.png`: Full A and B weight matrices as heatmaps.
- `weight_A_hists.png`, `weight_B_hists.png`: Per-hidden-unit distributions for A and B.

**Image datasets (mnist, cifar10, stl10, ising, xy, potts):**
- `test_vs_recon.png`: Side-by-side comparison of 10 test samples and their reconstructions.
- `test_vs_recon_hist.png`: Pixel value histograms for test vs. reconstructed samples.
- `test_vs_recon_h_hist.png`: Hidden unit activation histograms for test vs. reconstructed samples.
- `gen_images.png`: Grid of freely generated samples.
- `gen_hist.png`: Pixel value histograms of generated samples.
- `weight_images.png`: Weight matrix filters visualized as images.

**Ising / XY / Potts datasets:**
- `physical_properties.png`: Bar chart comparing energy per spin (E/N), heat capacity (C/N), magnetization (M/N), and susceptibility (χ/N) across test, reconstructed, and generated samples.

**Wind direction dataset:**
- `Test_recon_gen_angles.png`: Overlaid angle histograms for test, reconstructed, and generated samples.

**Protein dihedral dataset:**
- `phi_psi_angles.png`: φ and ψ marginal distributions for test, reconstructed, and generated samples.
- `ramachandran.png`: Ramachandran scatter plots for test, reconstructed, and generated samples.

### Notes
- `run_sample.py` must be run before `run_figures.py`. The `--n_samples` and `--k_gen` arguments select which saved sample file to load, matched against `outputs/samples/default_run/metadata.json`.
- If both `--n_samples` and `--k_gen` are omitted, the most recently saved sample file is used.