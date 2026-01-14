# WieLernenNeuronaleNetze (How Neural Networks Learn)

An educational research project exploring how neural networks learn, with a focus on **Information Bottleneck Theory** by Naftali Tishby.

## Overview

This repository contains 3 Jupyter notebooks:

### 1. Memorization (`Memorization.ipynb`)
Explores when DNNs "memorize data" vs. when they abstract and generalize. Demonstrates:
- The relationship between overfitting and memorization
- How regularization techniques (L2, Dropout) prevent memorization
- Uses CIFAR-10 dataset with randomly shuffled labels to show that NNs can memorize arbitrary mappings

### 2. Information Theory (`Information Theory.ipynb`)
Introduces information theory and the **Information Bottleneck Theory**. Shows how to:
- Examine neural networks for their information content
- Draw conclusions about learning success
- Understand how individual layers function

### 3. Mutual Information Calculation (`mutual_calculation.ipynb`)
A step-by-step implementation using a tiny DNN to calculate mutual information. Designed to be simple enough that each step can be verified manually.

## Requirements

- **Python**: 3.10+
- **PyTorch**: 2.9+
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended)

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/RoblabWh/WieLernenNeuronaleNetze
cd WieLernenNeuronaleNetze

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Launch Jupyter
jupyter lab
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Project Structure

```
WieLernenNeuronaleNetze/
├── pyproject.toml              # Project dependencies (PyTorch 2.9+)
├── Information Theory.ipynb    # Main information theory notebook
├── Memorization.ipynb          # Memorization vs generalization study
├── mutual_calculation.ipynb    # Step-by-step MI calculation
├── idnns/                      # Core library
│   ├── networks/               # Neural network implementations
│   │   ├── model.py           # PyTorch Model class with activation capture
│   │   ├── models.py          # CNN and MLP architectures
│   │   ├── network.py         # Training loop and utilities
│   │   └── utils.py           # Data loading (MNIST, CIFAR-10)
│   ├── information/            # Information theory calculations
│   │   ├── entropy_estimators.py        # k-NN entropy estimation
│   │   ├── mutual_info_estimation.py    # Variational MI estimation
│   │   └── information_process.py       # Main info calculation pipeline
│   └── plots/                  # Visualization utilities
└── data/                       # Datasets (MNIST, pre-computed matrices)
```

## Migration Note

This project was migrated from **TensorFlow 1.15** to **PyTorch 2.9** in January 2026. Key changes:
- Session-based execution → Eager execution
- `tf.placeholder` → Direct tensor inputs
- `tf.Variable` → `nn.Parameter` / `nn.Module`
- Custom activation extraction via forward hooks

Pre-trained models from the TensorFlow version are not compatible with the new PyTorch code.

## Credits and References

### Papers
- [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810) - Shwartz-Ziv & Tishby
- [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/abs/1503.02406) - Tishby & Zaslavsky
- [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530) - Zhang et al.
- [A Closer Look at Memorization in Deep Networks](https://arxiv.org/abs/1706.05394) - Arpit et al.

### Additional Resources
- [Quanta Magazine: New Theory Cracks Open the Black Box of Deep Learning](https://www.quantamagazine.org/new-theory-cracks-open-the-black-box-of-deep-learning-20170921/)
- [YouTube: Information Theory of Deep Learning (Tishby)](https://www.youtube.com/watch?v=pFWiauHOFpY)
- [YouTube: Deep Learning and Information Theory](https://www.youtube.com/watch?v=0Ey02HT_1Ho)

### Related Papers
- https://arxiv.org/abs/1611.05397
- https://arxiv.org/abs/1705.05363
- https://arxiv.org/abs/1707.01495
