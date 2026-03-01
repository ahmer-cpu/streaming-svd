# Streaming SVD

Research package for Streaming and Warm-started Randomized Singular Value Decomposition (rSVD).
This repository provides a framework for implementing and evaluating streaming SVD algorithms with support for warm-start initialization.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```

4. **(Optional) Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Project Structure

```
streaming-svd/
├── README.md                          # Project documentation
├── pyproject.toml                     # Project configuration and dependencies
├── .gitignore                         # Git ignore rules
│
├── src/
│   └── streaming_svd/                 # Main package
│       ├── __init__.py                # Package initialization
│       │
│       ├── algos/                     # SVD and rSVD algorithms
│       │   └── __init__.py
│       │
│       ├── data/                      # Data loading and preprocessing
│       │   └── __init__.py
│       │
│       ├── experiments/               # Experiment runners and benchmarks
│       │   └── __init__.py
│       │
│       └── utils/                     # Common utilities (logging, timing, I/O)
│           └── __init__.py
│
├── tests/                             # Unit tests
│   └── __init__.py
│
├── data/                              # Data directory
│   ├── raw/                           # Original data (not tracked in git)
│   ├── interim/                       # Intermediate processed data
│   └── processed/                     # Final processed data
│
├── results/                           # Experiment results
│   └── figures/                       # Generated figures and plots
│
├── notebooks/                         # Jupyter notebooks (exploratory)
│
├── scripts/                           # CLI scripts (if any)
│
└── docs/                              # Documentation (if any)
```

### Module Descriptions

- **`algos`**: Core SVD and randomized SVD algorithms. Place algorithm implementations here.
- **`data`**: Dataset utilities, loaders, and preprocessing functions. Place data I/O code here.
- **`experiments`**: Experiment runners, benchmarks, and evaluation scripts. Place benchmark code here.
- **`utils`**: Common utilities including timing, logging, and file I/O helpers.

## Usage

```python
from streaming_svd import algos, data, experiments, utils

# Import specific modules as needed
# from streaming_svd.algos import your_algorithm
# from streaming_svd.data import your_loader
```

## Tests

Run the test suite:
```bash
pytest
```

With coverage:
```bash
pytest --cov=streaming_svd
```

## Development

Format code with black:
```bash
black src tests
```

Sort imports with isort:
```bash
isort src tests
```

Type check with mypy:
```bash
mypy src
```

Lint with flake8:
```bash
flake8 src tests
```

## License

MIT

