from setuptools import setup, find_packages
import re

# Get version without importing the package
with open('app/version.py', 'r') as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in app/version.py")

setup(
    name="mlx-server",
    version=version,
    description="OpenAI-compatible server for MLX models",
    author="MLX Server Team",
    packages=find_packages(),
    install_requires=[
        "mlx-vlm",
        "fastapi",
        "uvicorn",
        "Pillow",
        "click",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ]
    },
    entry_points={
        "console_scripts": [
            "mlx-server=app.cli:cli",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 