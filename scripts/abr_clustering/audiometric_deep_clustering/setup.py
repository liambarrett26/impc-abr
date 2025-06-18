"""
Setup script for ContrastiveVAE-DEC audiometric phenotype discovery.

This package implements a novel deep learning approach for discovering
audiometric phenotypes in mouse genetic data from the IMPC.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
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
    author="IMPC ABR Clustering Research Team",
    author_email="", # Add email if needed
    description="ContrastiveVAE-DEC for audiometric phenotype discovery in mouse genetic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/impc/audiometric-deep-clustering",  # Update with actual repository
    project_urls={
        "Bug Reports": "https://github.com/impc/audiometric-deep-clustering/issues",
        "Source": "https://github.com/impc/audiometric-deep-clustering",
        "Documentation": "https://audiometric-deep-clustering.readthedocs.io/",
    },
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
    extras_require={
        "dev": dev_requirements,
        **optional_requirements
    },
    entry_points={
        "console_scripts": [
            "audiometric-train=scripts.train:main",
            "audiometric-evaluate=scripts.evaluate:main", 
            "audiometric-infer=scripts.infer:main",
            "audiometric-visualize=scripts.visualize_results:main",
        ],
    },
    package_data={
        "audiometric_deep_clustering": [
            "config/*.yaml",
            "config/*.yml",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    
    # Keywords for PyPI search
    keywords=[
        "deep learning",
        "clustering", 
        "variational autoencoder",
        "contrastive learning",
        "bioinformatics",
        "phenotype discovery",
        "mouse genetics",
        "audiometry",
        "IMPC"
    ],
    
    # Additional metadata
    platforms=["any"],
    license="MIT",
    
    # Testing configuration
    test_suite="tests",
    tests_require=dev_requirements,
    
    # Custom commands
    cmdclass={},
    
    # Configuration for different installation scenarios
    options={
        "build_scripts": {
            "executable": "/usr/bin/env python",
        },
    },
)

# Post-installation message
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ContrastiveVAE-DEC Installation Complete                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Thank you for installing ContrastiveVAE-DEC for audiometric phenotype      ║
║  discovery! This package provides state-of-the-art deep learning tools      ║
║  for clustering and analyzing audiometric data from mouse genetic studies.  ║
║                                                                              ║
║  Quick Start:                                                                ║
║    1. Train a model: audiometric-train --config config/training_config.yaml ║
║    2. Evaluate results: audiometric-evaluate --checkpoint path/to/model.ckpt║
║    3. Generate visualizations: audiometric-visualize --embeddings path.npz  ║
║                                                                              ║
║  Documentation: See README.md and CLAUDE.md for detailed usage instructions ║
║  Configuration: Check config/ directory for example configurations          ║
║  Examples: Explore notebooks/ directory for usage examples                  ║
║                                                                              ║
║  For support and issues: https://github.com/impc/audiometric-deep-clustering║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

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