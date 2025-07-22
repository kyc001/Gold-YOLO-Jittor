#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gold-yolo-jittor",
    version="1.0.0",
    author="新芽第二阶段",
    author_email="",
    description="Gold-YOLO implementation using Jittor framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/Gold-YOLO_jittor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gold-yolo-train=scripts.train:main",
            "gold-yolo-eval=scripts.evaluate:main",
            "gold-yolo-infer=scripts.inference:main",
        ],
    },
)
