from setuptools import find_packages, setup

setup(
    name="abr_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy", "matplotlib", "scikit-learn"],
)
