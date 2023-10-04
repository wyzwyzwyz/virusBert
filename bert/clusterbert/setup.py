'''
Author: your name
Date: 2020-12-08 04:05:33
LastEditTime: 2022-03-24 16:51:00
LastEditors: Yang jin
Description: In User Settings Edit
FilePath: /BERT-pytorch/setup.py
'''
from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys

__version__ = "2.0"

with open("requirements.txt") as f:
    require_packages = [line[:-1] for line in f]

with open("README.md", "r" ) as f:
    long_description = f.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != __version__:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, __version__
            )
            sys.exit(info)


setup(
    name="clusterbert",
    version=__version__,
    author='Jin Yang,Zhijie Cai',
    author_email='jin.yang@siat.ac.cn',
    packages=find_packages(),
    install_requires=require_packages,
    url="https://github.com/",
    description="Pretrained BERT based multi-binary classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'bibert = bibert.__main__:train',
            'bibert-vocab = bibert.dataset.vocab:build' 
        ]
    },
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
