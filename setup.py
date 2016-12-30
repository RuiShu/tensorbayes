"""Setup file for tensorbayes

For easy installation and uninstallation, do the following.
INSTALL:
python setup.py install --record files.txt
UNINSTALL:
cat files.txt | xargs rm -r
Pip installation will be made available soon.
"""

from setuptools import setup, find_packages
import os

setup(
    name="tensorbayes",
    version="0.1.0",
    author="Rui Shu",
    author_email="ruishu@stanford.edu",
    url="http://www.github.com/RuiShu/tensorbayes",
    license="MIT",
    description="Deep Variational Inference in TensorFlow",
    install_requires = ['numpy'],
    extras_require={
        'notebook': ['jupyter']
    },
    packages=find_packages()
)
