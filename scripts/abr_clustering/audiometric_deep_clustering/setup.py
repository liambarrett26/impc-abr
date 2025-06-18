"""
Setup script for ContrastiveVAE-DEC audiometric phenotype discovery.

This package implements a novel deep learning approach for discovering
audiometric phenotypes in mouse genetic data from the IMPC.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
long_description = "ContrastiveVAE-DEC for audiometric phenotype discovery"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "umap-learn>=0.5.3",
        "statsmodels>=0.13.0"
    ]

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.971",
    "pre-commit>=2.20.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.15.0",
    "ipywidgets>=8.0.0"
]

# Optional dependencies for different use cases
optional_requirements = {
    "gpu": [
        "torch>=2.0.0+cu118",  # CUDA 11.8 version
    ],
    "experiment_tracking": [
        "wandb>=0.13.0",
        "tensorboard>=2.10.0",
        "mlflow>=1.28.0"
    ],
    "hyperopt": [
        "optuna>=3.0.0",
        "ray[tune]>=2.0.0"
    ],
    "distributed": [
        "torch-distributed>=0.1.0"
    ],
    "all": [
        "wandb>=0.13.0",
        "tensorboard>=2.10.0",
        "mlflow>=1.28.0",
        "optuna>=3.0.0",
        "ray[tune]>=2.0.0"
    ]
}

setup(
    name="audiometric-deep-clustering",
    version="1.0.0",
    author="Liam Barrett",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    package_data={
        "audiometric_deep_clustering": [
            "config/*.yaml",
            "config/*.yml",
        ],
    }
)

# Verification that key dependencies are available
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} detected")
    if torch.cuda.is_available():
        print(f"✓ CUDA {torch.version.cuda} available with {torch.cuda.device_count()} GPU(s)")
    else:
        print("! CUDA not available - will use CPU for training")
except ImportError:
    print("! PyTorch not found - please install manually if setup failed")

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__} detected")
except ImportError:
    print("! Scikit-learn not found - some features may not work")

try:
    import plotly
    print(f"✓ Plotly {plotly.__version__} detected")
except ImportError:
    print("! Plotly not found - interactive visualizations will not work")

print("\nInstallation verification complete. Ready to discover audiometric phenotypes!")