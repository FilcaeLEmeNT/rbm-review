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
│   ├── rbm_binary.ipynb
│   ├── RBM_exponential.ipynb
│   ├── RBM_Gaussian.ipynb
│   ├── RBM_multinomial.ipynb
│   └── rbm_vonmises.ipynb
├── outputs
│   ├── checkpoints
│   ├── figures
│   └── history
├── pyproject.toml
├── README.md
├── requirements-lock.txt
├── requirements.txt
├── scripts
│   └── run_train.py
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
```

### Description
- `configs/` – YAML configuration files for training hyperparameters and setup
- `data/` – Dataset directory for data storage (raw and processed)
- `notebooks/` – Jupyter notebooks demonstrating each RBM architecture.
- `outputs/` – Stores training outputs, including checkpoints, figures, and logs.
- `scripts/` – Excecutrion scripts for training and evaluation.
- `src/data/` – Data loading utilities.
- `src/models/` – Implementations of different RBM architectures.
- `src/training/` – Training loops and optimization code.
- `src/utils/` – Utility functions for reading config files and device management (CPU/GPU selection).

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

## 🚗 Running Training

Example training script:

```bash
python scripts/run_train.py --config configs/default.yaml    # Linux / Mac

# python scripts\run_train.py --config configs\default.yaml  # Windows
``` 
or
```bash
python -m scripts.run_train --config configs/default.yaml   # Linux / Mac

# python -m scripts.run_train --config configs\default.yaml # Windows
``` 

## 🚗 Configuration

The configs directory contains YAML files that can are used as an argument for specifying parameters and hyperparameters for training, used like:
```bash
python scripts/run_train.py --config configs/default.yaml
``` 