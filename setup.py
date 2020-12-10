# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="femder-gutoalvim", # Replace with your own username
    version="0.0.1",
    author="Luiz Augusto Alvim",
    author_email="luiz.alvim@eac.ufsm.br",
    description="A simple acoustic FEM package ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gutoalvim/femder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)