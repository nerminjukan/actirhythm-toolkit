"""Setup script for the actirhythm-toolkit package."""
from setuptools import setup, find_packages

setup(
    name="actirhythm-toolkit",
    version="0.2.0",
    description="Reproducible accelerometer analysis pipeline for circadian and behavioral rhythm studies",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nermin Jukan",
    url="https://github.com/nerminjukan/actirhythm-toolkit",
    project_urls={
        "Source": "https://github.com/nerminjukan/actirhythm-toolkit",
    },
    license="GPL-3.0-or-later",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "pyarrow>=5.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "hmmlearn>=0.2.7",
        "pyyaml>=5.4.0",
        "loguru>=0.5.3",
    ],
    extras_require={
        "ml": [
            "xgboost>=1.5.0",
            "pomegranate>=0.14.8",
            "CosinorPy>=1.1",
        ],
        "glmm": [
            "pymer4>=0.7.0",
            "polars>=1.0.0",
            "rpy2>=3.6.0",
            "great-tables>=0.23.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "nbformat>=5.1.0",
            "tqdm>=4.62.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "python-dotenv>=0.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "actirhythm=src.cli:main",
        ],
    },
    python_requires=">=3.8",
)
