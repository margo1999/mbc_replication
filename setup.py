"""
Installs the "mbc_network" library.
"""
from setuptools import setup, find_packages

setup(
    name="mbc_network",
    version="1.0",
    description="PyNEST implementation of the network proposed by Maes et al. 2020",
    author="Jette Oberlaender, Younes Bouhadjar",
    author_email="y.bouhadjar@fz-juelich.de",
    url='https://github.com/margo1999/mbc_replication',
    packages=find_packages()
)
