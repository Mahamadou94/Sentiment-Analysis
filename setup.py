#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    reqs = [req.rstrip() for req in f]


setup(
    name="pp_classifier",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "preprocess = pp_classifier.cli.preprocess:main",
            "translate = pp_classifier.cli.translate:main",
            "train = pp_classifier.cli.train:main",
            "predict = pp_classifier.cli.predict:main"
        ]
    },
    python_requires=">=3.8",
    install_requires=reqs
)