# 🔥 PyTorch Experimentation Repo

A clean, well-organized repository for experimenting with PyTorch ideas and concepts.

## 🚀 Getting Started

This project uses [UV](https://github.com/astral-sh/uv) for fast, modern Python package management.

### Prerequisites

Install UV if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd TorchExperimenting

# Install dependencies (UV will automatically create a venv)
uv sync

# Run Python scripts
uv run python experiments/your_experiment.py

# Launch Jupyter
uv run jupyter lab
```

## 📁 Project Structure

```
TorchExperimenting/
├── experiments/        # Python modules for experiments
│   └── __init__.py
├── notebooks/          # Jupyter notebooks for interactive work
├── tests/             # Unit tests
├── pyproject.toml     # Project dependencies
└── README.md
```

## 📦 Installed Packages

- **torch** - PyTorch deep learning framework
- **torchvision** - Computer vision utilities
- **numpy** - Numerical computing
- **matplotlib** - Plotting and visualization
- **jupyter** - Interactive notebooks
- **tensorboard** - Experiment tracking
- **tqdm** - Progress bars

## 💡 Usage Examples

### Quick Script
```python
import torch
import torch.nn as nn

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Your experiment here
model = nn.Linear(10, 2).to(device)
```

### Running Scripts
```bash
uv run python experiments/my_experiment.py
```

### Jupyter Notebooks
```bash
uv run jupyter lab
```

## 🔧 Adding Dependencies

```bash
# Add a new package
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Remove a package
uv remove <package-name>
```

## 📊 TensorBoard

Track your experiments:
```bash
uv run tensorboard --logdir=runs
```

## 🧪 Testing

```bash
uv run pytest tests/
```

## 📝 Notes

- Virtual environment is automatically managed in `.venv/`
- Dependencies are locked in `uv.lock` for reproducibility
- PyTorch with MPS support (Apple Silicon) or CUDA (if available)

## 🤝 Contributing

This is a personal experimentation repo, but feel free to fork and adapt for your own projects!

## 📄 License

MIT License - Feel free to use and modify as needed.

