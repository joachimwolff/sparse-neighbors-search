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
import time 
__author__ = "Joachim Wolff"
__contact__ = "wolffj@informatik.uni-freiburg.de"
__copyright__ = "Copyright 2015, Joachim Wolff"
__credits__ = ["Milad Miladi", "Fabrizio Costa"]
__license__ = "No License"
__date__ = time.strftime("%d/%m/%Y")
__version_ = "0.1.dev"

from setuptools import setup, find_packages
from distutils.core import Extension
import platform
import sys

sources_list = ['neighborsMinHash/computation/interface/minHash_PythonInterface.cpp', 'neighborsMinHash/computation/minHash.cpp', 
                'neighborsMinHash/computation/minHashBase.cpp', 'neighborsMinHash/computation/inverseIndex.cpp',
                 'neighborsMinHash/computation/inverseIndexStorageBloomierFilter.cpp' , 'neighborsMinHash/computation/inverseIndexStorageUnorderedMap.cpp',
                 'neighborsMinHash/computation/bloomierFilter/bloomierFilter.cpp', 'neighborsMinHash/computation/bloomierFilter/bloomierHash.cpp',
                 'neighborsMinHash/computation/bloomierFilter/encoder.cpp','neighborsMinHash/computation/bloomierFilter/orderAndMatchFinder.cpp']
depends_list = ['neighborsMinHash/computation/minHash.h', 'neighborsMinHash/computation/minHashBase.h', 'neighborsMinHash/computation/inverseIndex.h',
         'neighborsMinHash/computation/typeDefinitions.h', 'neighborsMinHash/computation/parsePythonToCpp.h', 'neighborsMinHash/computation/sparseMatrix.h',
          'neighborsMinHash/computation/inverseIndexStorage.h',
                 'neighborsMinHash/computation/inverseIndexStorageBloomierFilter.h' , 'neighborsMinHash/computation/inverseIndexStorageUnorderedMap.h',
                 'neighborsMinHash/computation/bloomierFilter/bloomierFilter.h', 'neighborsMinHash/computation/bloomierFilter/bloomierHash.h',
                 'neighborsMinHash/computation/bloomierFilter/encoder.h','neighborsMinHash/computation/bloomierFilter/orderAndMatchFinder.h']
if "--openmp" in sys.argv:
    module1 = Extension('_minHash', sources = sources_list, depends = depends_list,
         define_macros=[('OPENMP', None)], extra_link_args = ["-lm", "-lrt","-lgomp"], 
        extra_compile_args=["-fopenmp", "-O3", "-std=c++11"])#, include_dirs=['/home/wolffj/Software/boost_1_59_0', '/home/wolffj/Software/mtl4'])
# extra_link_args=(['-Wl,--no-undefined'])
elif platform.system() == 'Darwin' or "--noopenmp" in sys.argv:
    module1 = Extension('_minHash', sources = sources_list, depends = depends_list, 
        extra_compile_args=["-O3", "-std=c++11"])

else:
    module1 = Extension('_minHash', sources = sources_list, depends = depends_list,
        define_macros=[('OPENMP', None)], extra_link_args = ["-lm", "-lrt","-lgomp"],
         extra_compile_args=["-fopenmp", "-O3", "-std=c++11"])
if "--openmp" in sys.argv:
    sys.argv.remove("--openmp")
if "--noopenmp" in sys.argv:
    sys.argv.remove("--noopenmp")


setup (name = 'neighborsMinHash',
        author = 'Joachim Wolff',
        author_email = 'wolffj@informatik.uni-freiburg.de',
        url='https://github.com/joachimwolff/minHashNearestNeighbors',
        license='LICENSE',
        description='An approximate computation of nearest neighbors based on locality sensitive hash functions.',
        long_description=open('README.md').read(),
        install_requires=[
        "numpy >= 1.8.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.16.0",],
        ext_modules = [module1],
        packages=['neighborsMinHash',
                    'neighborsMinHash.neighbors',
                    'neighborsMinHash.util',
                    'neighborsMinHash.clustering',
                    #  'neighborsMinHash.computation',
                ],
        platforms = "Linux, Mac OS X",
        version = '0.1.dev'
        )
