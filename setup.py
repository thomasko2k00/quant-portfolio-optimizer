"""
Setup configuration for portfolio optimizer package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quant-portfolio-optimizer",
    version="1.0.0",
    author="Quant Finance Expert",
    author_email="your.email@example.com",
    description="Advanced portfolio optimization with Ledoit-Wolf, Black-Litterman, and ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quant-portfolio-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "cvxpy>=1.4.0",
        "yfinance>=0.2.28",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "ml": ["tensorflow>=2.13.0", "keras>=2.13.0"],
        "api": ["fastapi>=0.104.0", "uvicorn>=0.24.0"],
        "dashboard": ["streamlit>=1.28.0"],
        "dev": ["pytest>=7.4.0", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "portfolio-optimize=portfolio_optimizer:main",
        ],
    },
)
