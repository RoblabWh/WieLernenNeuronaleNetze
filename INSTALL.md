# Installation

## Requirements

- **Python**: 3.10 or higher
- **GPU**: CUDA-compatible GPU recommended (but not required)

## With uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that replaces pip and venv.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/RoblabWh/WieLernenNeuronaleNetze
cd WieLernenNeuronaleNetze

# 3. Create virtual environment
uv venv

# 4. Activate the environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 5. Install the project
uv pip install -e .

# 6. (Optional) Install development dependencies
uv pip install -e ".[dev]"
```

## With pip

If you prefer traditional pip:

```bash
# 1. Clone the repository
git clone https://github.com/RoblabWh/WieLernenNeuronaleNetze
cd WieLernenNeuronaleNetze

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate the environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 4. Install the project
pip install -e .
```

## Running Notebooks

After installation, launch JupyterLab:

```bash
jupyter lab
```

Then open one of the notebooks:
- `Information Theory.ipynb` - Main information theory analysis
- `Memorization.ipynb` - Memorization vs generalization study
- `mutual_calculation.ipynb` - Step-by-step MI calculation

## GPU Support

PyTorch will automatically use CUDA if available. To verify:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Troubleshooting

### PyTorch CUDA Issues
If PyTorch doesn't detect your GPU, you may need to install a CUDA-specific version:

```bash
# For CUDA 12.x
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues
If you encounter out-of-memory errors during training or information calculation, try reducing the batch size in the notebook parameters.
