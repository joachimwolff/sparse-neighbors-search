#!/usr/bin/python
#! /usr/bin/python
# Copyright 2015 Joachim Wolff
# Master Thesis
# Tutors: Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau
__author__ = "Joachim Wolff"
__contact__ = "wolffj@informatik.uni-freiburg.de"
__copyright__ = "Copyright 2015, Joachim Wolff"
__credits__ = ["Milad Miladi", "Fabrizio Costa"]
__license__ = "No License"
__date__ = "05.08.2015"
__version_ = "0.1.dev"

from setuptools import setup, find_packages
from distutils.core import Extension
import platform
import sys

module1 = Extension('_convertDataToCpp', sources = ['util/convertDataToCpp.cpp'],         
                                        extra_compile_args=["-O3", "-std=c++11"])


setup (name = 'convertDataToCpp',
        author = 'Joachim Wolff',
        author_email = 'wolffj@informatik.uni-freiburg.de',
        license='LICENSE',
        description='Convert data from python csr to three files name_instances, name_features, name_data. These three can be taken as input for vectors.',
        # long_description=open('README.md').read(),
        # install_requires=[
        # "numpy >= 1.8.0",
        # "scipy >= 0.14.0",
        # "scikit-learn >= 0.16.0",],
        ext_modules = [module1],
        packages=['util',
                    # 'neighborsMinHash.neighbors',
                    # 'neighborsMinHash.computation',
		    # 'neighborsMinHash.util',
            # 'neighborsMinHash.clustering',
                ],
        platforms = "Linux",
        version = '0.1.dev'
        )
