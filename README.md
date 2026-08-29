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
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
│   ├── resume_train.py
│   ├── run_train.py
|   ├── run_sample.py
|   └── run_figures.py
└── src
    ├── data
    │   ├── data_loader.py
    │   ├── datasets.py
    ├── models
    │   ├── rbm_binary.py
    │   ├── rbm_exponential.py
    │   ├── rbm_gaussian.py
    │   ├── rbm_multinomial.py
    │   └── rbm_vonmises.py
    ├── training
    │   └── training.py
    └── utils
        ├── checkpoint.py
        ├── config.py
        ├── device.py
        ├── multinomial.py
        ├── physics.py
        ├── sweep.py
```

### Description
- `configs/` – YAML configuration files for training hyperparameters and setup
- `data/` – Dataset directory for data storage (raw and processed)
- `notebooks/` – Jupyter notebooks demonstrating each RBM architecture.
- `outputs/` – Stores training outputs, including checkpoints, figures, and model samples.
- `scripts/` – Excecutrion scripts for training and evaluation.
- `src/data/` – Data loading utilities and custom Dataset classes.
- `src/models/` – Implementations of different RBM architectures.
- `src/training/` – Training loops and optimization code.
- `src/utils/` – Utility functions for config loading, device management, checkpoint saving/loading, transformations for multinomial datasetes, and physical observables.

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

---
Sweep configuration files can be used along side normal configuration files to run multiple training with different parameters consecutively, used like:
```bash
python scripts/run_train.py --config configs/potts.yaml --sweep configs/T_sweep.yaml
```
If one wants to sweep through different parameters, T, lr, and pcd for example, they can format a sweep configuration file like:
```
data.T: [0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000, 1.050, 1.100, 1.150, 1.200]
training.lr: [0.1, 0.01, 0.001]
training.pcd: [True, False]
```
This will run 13 x 3 x 2 consecutive trainings. Notice a '.' is used to navigate the dictionary. The values in the sweep configuration file will overwrite the values in the normal configuration file.

A suffix will be added to the run_name specified in the config file to distinguish the different runs, formatted like: "potts_T=0.6_lr=0.1_pcd=True" for the first run in the example above. The parameters in the run_name will follow the same order as in the sweep configuration file.

## 🗂️ Datasets

In the config file, data.data_type can be set to 'mnist', 'stl10', 'cifar10', 'ising', 'xy', 'potts', 'wind_dir', or 'protein'.
- **For 'mnist', 'stl10', and 'cifar10'**: The dataset will be automatically downloaded to the directory specified by data.data_dir in the config file.
- **For 'ising', 'xy', 'potts'**: Download the dataset from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19712829.svg)](https://doi.org/10.5281/zenodo.19712829) and put it inside the directory specified by data.data_dir in the config file.
- **For 'wind_dir'**: Download the dataset from [this website](https://www.ndbc.noaa.gov/download_data.php?filename=42503e2023.txt.gz&dir=data/historical/drift/) and put it inside the directory specified by data.data_dir in the config file. The point of this is to provide an example of learning a dataset with one angular visible unit, wind direction.
- **For 'protein'**: A Sidechain Net ([Link to Github](https://github.com/jonathanking/sidechainnet)) dataset will be automatically downloaded to the directory specified by data.data_dir in the config file. Moreover, two dihedral angles, phi and psi are automatically parsed to create a dataset that is loaded while training and sampling. The dataset may take a while to download.

### Model-data compatibility
The datasets are only compatible with certain types of RBMs.
```
VALID_MODEL_FOR_DATA = {
    "mnist": ["binary", "exponential", "gaussian", "multinomial"],
    "cifar10": ["binary", "exponential", "gaussian", "multinomial"],
    "stl10": ["binary", "exponential", "gaussian", "multinomial"],
    "ising": ["binary"],
    "potts": ["multinomial"],
    "xy": ["vonmises"],
    "wind_dir": ["vonmises"],
    "protein": ["vonmises"],
}
```

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
- `--config`: Path to the configuration file that specifies the dataset, model architecture, training hyperparameters, etc.
- `--sweep`: Path to the sweep configuration file that specifies parameter overwrites for multiple consecutive runs.

## 🚩 Checkpoints

Upon running training, a checkpoint is saved to `outputs/checkpoints/default_run/`, assuming output.base_dir is set to 'outputs' and run_name is set to 'default_run' in the configs file. It contains everything needed to reconstruct the model without any additional arguments:

```python
{
    "epoch": int,
    "model_state": dict,  # model weights — e.g. W, b_v, b_h
    "optimizer_state": dict,  # None when an optimizer is not used.
    "config": dict,  # architecture + hyperparameters
    "history": dict,  # train loss/energy history
}
```

Because the config is embedded, downstream scripts can take `--checkpoint` as their only required argument. No separate `--config` needed:

```bash
python scripts/run_eval.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt

python scripts/run_sample.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt

python scripts/run_figures.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt
```

Note: `--config` can be used as an alternative argument to search for the checkpoint file path automatically, which may be more convenient when the directory structure is more complicated.

To load a checkpoint and the model:

```python
from utils.checkpoint import load_checkpoint

model, ckpt = load_checkpoint(ckpt_path, device)
```

## 🚦 Resume Training
Example prompt:

```bash
python scripts/resume_train.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt --n_epochs 100  # Linux / Mac

# python scripts\resume_train.py --checkpoint outputs\checkpoints\default_run\checkpoint.pt  # Windows
``` 
### Arguments
- `--checkpoint`: Path to the checkpoint file generated by run_train.py.
- `--sweep`: Combine with --config to resume all runs generated from that config/sweep combination.
- `--config`: Path to config file. Used to infer the checkpoint path from output.basedir and output.run_name. Use as an alternative to --checkpoint, or combine with --sweep to resume all runs generated from that config/sweep combination.
- `--n_epochs`: Number of epochs to train, resuming from a checkpoint. If omitted, a default value of 100 is used.

### Outputs
This overwrites the checkpoint file from which the file was generated from and updates the epoch value in the checkpoint as well as the n_epochs value in the config file embedded in the checkpoint.

## 🏭 Sampling

Example prompt:

```bash
python scripts/run_sample.py --checkpoint outputs/checkpoints/default_run/checkpoint.pt --k_gen 1000 --n_samples 8192  # Linux / Mac

# python scripts\run_sample.py --checkpoint outputs\checkpoints\default_run\checkpoint.pt --k_gen 1000 --n_samples 8192  # Windows
``` 

### Arguments
- `--checkpoint`: Path to the checkpoint file generated by run_train.py.
- `--sweep`: Combine with --config to sample from all runs generated from that config/sweep combination.
- `--config`: Path to config file. Used to infer the checkpoint path from output.basedir and output.run_name. Use as an alternative to --checkpoint, or combine with --sweep to sample from all runs generated from that config/sweep combination.
- `--k_gen`: Amount of k steps used to generate samples from a randomized tensor. Defaults to 1000.
- `--n_samples`: Amount of samples generated from the script. Defaults to 8192.

### Outputs
These files are saved to `outputs/samples/default_run/` (the path depends on the output settings in the config file).
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
- `--sweep`: Combine with --config to create figures from all runs generated from that config/sweep combination.
- `--config`: Path to config file. Used to infer the checkpoint path from output.basedir and output.run_name. Use as an alternative to --checkpoint, or combine with --sweep to create figures from all runs generated from that config/sweep combination.
- `--k_gen`: Selects which saved sample file to use by matching `k_gen`. If omitted, the most recent sample matching `n_samples` is used.
- `--n_samples`: Selects which saved sample file to use by matching `n_samples`. If omitted, the most recent sample matching `k_gen` is used.
- `--skip_weights`: When used skips all figures involving the visualizing the weights.

### Outputs
These files are saved to `outputs/figures/default_run/` (the path depends on the output parameters in the config file).

**Warning: Below is not updated.**

**All models:**
- `training_curve.png`: Energy of data and model configurations, energy difference, reconstruction MSE, and reconstruction cross-entropy over training epochs.
- `weight_hist.png`: Distribution of weight matrix values.
- `weight.png`: Full weight matrix as a heatmap.
- `weight_hists.png`: Per-hidden-unit weight distributions.

**Von Mises RBM (has two weight matrices A and B instead of W):**
- `weight_A_hist.png`, `weight_B_hist.png`: Distributions of the A and B weight matrices.
- `weight_A.png`, `weight_B.png`: Full A and B weight matrices as heatmaps.
- `weight_A_hists.png`, `weight_B_hists.png`: Per-hidden-unit distributions for A and B.

**Multinomial RBM (W is divided into W_i, where i={0, 1, ..., n_class-1}):**
- `weight_q=0_hist.png`, `weight_q=1_hist.png`, ...: Distribution of weight matrix values for each class.
- `weight_q=0.png`, `weight_q=1.png`, ...: Weight matrices divided by classes as a heatmap .
- `weight_q=0_hists.png`, `weight_q=1_hists.png`, ...: Per-hidden-unit weight distributions for each class.

**Image datasets (mnist, cifar10, stl10, ising, xy, potts):**
- `test_vs_recon.png`: Side-by-side comparison of 10 test samples and their reconstructions.
- `test_vs_recon_hist.png`: Pixel value histograms for test vs. reconstructed samples.
- `test_vs_recon_h_hist.png`: Hidden unit activation histograms for test vs. reconstructed samples.
- `persistent_batch.png`: If pcd is used, shows the persistent batch at the end of training.
- `gen_images.png`: Grid of freely generated samples.
- `gen_hist.png`: Pixel value histograms of generated samples.
- `weight_images.png`: Weight matrix filters visualized as images. For Multinomial RBM: `weight_q=0_images.png`, ... as in above.

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

## 📌 License

This project is licensed under the MIT License. See the `LICENSE` file for details.