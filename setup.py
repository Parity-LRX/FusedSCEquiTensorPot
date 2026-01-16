"""Setup script for molecular_force_field package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if exists
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

setup(
    name="molecular_force_field",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for molecular modeling with E3NN-based neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/molecular_force_field",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "e3nn>=0.5.0",
        "torch-scatter>=2.0.9",
        "torch-cluster>=1.6.0",
        "ase>=3.22.0",
        "tqdm>=4.62.0",
        "h5py>=3.6.0",
        "matplotlib>=3.5.0",
    ],
    entry_points={
        "console_scripts": [
            "mff-train=molecular_force_field.cli.train:main",
            "mff-evaluate=molecular_force_field.cli.evaluate:main",
            "mff-preprocess=molecular_force_field.cli.preprocess:main",
            "mff-lammps=molecular_force_field.cli.lammps_interface:main",
        ],
    },
)