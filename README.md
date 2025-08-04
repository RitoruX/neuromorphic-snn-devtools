# Neuromorphic SNN Development Tools

This project explores how to build Spiking Neural Networks (SNNs) for neuromorphic hardware, using modern software tools. The main goal is to compare frameworks, learn what works best, and share practical tips for others interested in this field.

## What’s Inside

This repository is part of an MSc dissertation at the University of Strathclyde. It includes code, experiments, and notes about working with SNNs on neuromorphic processors.

## Goals

- Try out different neuromorphic frameworks
- Build and train SNNs using real datasets
- Compare performance and usability
- Share what works, what doesn’t, and why
- Make it easy for others to follow along

## Frameworks Used

### Intel Lava

- Location: `frameworks/lava/notebooks/`
- What it does: Open-source tools for building and running SNNs on Intel’s neuromorphic hardware
- Example notebooks:
  - `00_lava_init.ipynb`: Checks the environment and basic setup
  - `01_slayer_lavadl_nmnist.ipynb`: Training SNNs with SLAYER and Lava-DL
  - Other notebooks: Experiments with CIFAR10-DVS and DVS Gesture datasets

### BrainChip Akida

- Location: `frameworks/akida/notebooks/`
- What it does: Tools for building SNNs compatible with BrainChip’s Akida hardware
- Example notebooks:
  - `01_akida_cifar10_dvs.ipynb`: Training and evaluating Akida models on CIFAR10-DVS
  - `02_akida_cifar10_dvs_sequential.ipynb`: Sequential training and evaluation

## Folder Structure

```
neuromorphic-snn-devtools/
├── README.md
├── LICENSE
├── envs/                  # Environment configs for Lava and Akida
├── frameworks/
│   ├── akida/             # Akida models, notebooks, images
│   └── lava/              # Lava models, notebooks, images
├── shared/                # Common code (datasets, training, monitoring)
└── datasets/              # DVS Gesture, CIFAR10-DVS, NMNIST data
```

## Getting Started

1. **Clone the repo**
   ```bash
   git clone <repository-url>
   cd neuromorphic-snn-devtools
   ```

2. **Set up your environment**
   ```bash
   conda env create -f envs/akida.yaml   # or envs/loihi.yaml for Lava
   conda activate akida-env              # or loihi-env for Lava
   ```


## Experiments

- SNN training and evaluation on DVS Gesture and CIFAR10-DVS datasets
- Model conversion and benchmarking for Lava and Akida
- Early stopping, system monitoring, and learning curve plotting
- Troubleshooting common issues (shape, dtype, memory, plotting)

---
