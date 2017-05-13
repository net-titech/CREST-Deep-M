# coding: utf-8

from setuptools import setup, find_packages

with open("README.rst") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name='crestdeep',
    version='0.1.0',
    description="Deep compression package for caffe models",
    long_description=readme,
    author="Murata research group",
    author_email="group@net-titech.org",
    url="https://net-titech.github.io/",
    license=license
    packages=find_packages(exclude=('tests', 'docs'))
)
