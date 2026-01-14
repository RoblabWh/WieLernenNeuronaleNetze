# CLAUDE.md - Project Context

This file provides context about the WieLernenNeuronaleNetze project for AI assistants.

## Project Overview

**WieLernenNeuronaleNetze** ("How Neural Networks Learn") is an educational research project studying neural network learning dynamics using Information Bottleneck Theory by Naftali Tishby.

The project analyzes:
- Mutual information flow through network layers during training
- The distinction between memorization and generalization
- How regularization affects learning behavior

## Tech Stack

- **Python**: 3.10+
- **Deep Learning**: PyTorch 2.9+
- **Data Science**: NumPy, SciPy, Matplotlib, Pandas
- **Package Manager**: uv
- **Notebooks**: JupyterLab

## Key Directories

```
idnns/networks/     # Neural network implementations (PyTorch)
idnns/information/  # Mutual information calculation (mostly NumPy)
idnns/plots/        # Visualization utilities
data/               # MNIST data and pre-computed matrices
models/             # Saved model checkpoints
```

## Important Files

| File | Purpose |
|------|---------|
| `idnns/networks/model.py` | Main `Model` class (PyTorch nn.Module) with activation capture |
| `idnns/networks/network.py` | Training loop with `train_network()` function |
| `idnns/information/information_process.py` | Main entry point for MI calculation |
| `idnns/information/entropy_estimators.py` | k-NN based entropy estimation (pure NumPy) |
| `idnns/networks/information_network.py` | High-level orchestration class |

## Architecture Notes

### Activation Capture
The `Model` class captures hidden layer activations during forward pass:
```python
# Activations stored in model._hidden_activations after forward()
model(x)
activations = model.hidden_layers  # List of layer outputs
```

### Information Calculation
The information calculation pipeline:
1. `get_information()` in `information_process.py` - main entry point
2. `calc_information_for_epoch()` - per-epoch calculation
3. `calc_information_for_layer()` - per-layer MI via binning
4. `estimate_IY_by_network()` - variational MI estimation with decoder network

### Data Loading
MNIST is loaded via torchvision:
```python
from idnns.networks.utils import load_data
data_sets = load_data('data/MNIST')  # Returns object with .data and .labels
```

## Common Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# Run Jupyter
jupyter lab

# Run tests (if available)
pytest tests/
```

## Development Guidelines

1. **Pure NumPy for MI**: The information calculation code (`entropy_estimators.py`, `mutual_information_calculation.py`) is pure NumPy - keep it that way for clarity

2. **Activation Capture**: Use `model.hidden_layers` after a forward pass to get layer activations as a list

3. **Device Handling**: The code auto-detects CUDA:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

4. **Training Data Format**: Data should be numpy arrays with shape `(N, features)` for data and `(N, classes)` for one-hot labels

## Known Limitations

- Pre-trained `.model` files are from TensorFlow and won't load in PyTorch
- The `mutual_calculation.ipynb` notebook may need manual updates to work with PyTorch
- GPU memory can be an issue with large batch sizes for activation extraction

## Testing Changes

After making changes:
1. Run `Information Theory.ipynb` end-to-end
2. Verify information plane plots generate correctly
3. Check that MI values are in reasonable ranges (0 to H(Y) typically)
