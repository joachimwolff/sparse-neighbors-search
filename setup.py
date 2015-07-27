#!/usr/bin/python
#! /usr/bin/python
# Copyright 2015 Joachim Wolff
# Master Project
# Tutors: Milad Miladi, Fabrizio Costa
# Summer semester 2015
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau

from distutils.core import setup, Extension
import platform
import sys

if "--openmp" in sys.argv:
    module1 = Extension('_hashUtility', sources = ['lib/src_c/hashUtility.cpp'], define_macros=[('OPENMP', None)], extra_link_args = ["-lm", "-lrt","-lgomp"], extra_compile_args=["-fopenmp", "-O3"])
elif platform.system() == 'Darwin' or "--noopenmp" in sys.argv:
    module1 = Extension('_hashUtility', sources = ['lib/src_c/hashUtility.cpp'], extra_compile_args=["-O3"])
else:
    module1 = Extension('_hashUtility', sources = ['lib/src_c/hashUtility.cpp'],  define_macros=[('OPENMP', None)], extra_link_args = ["-lm", "-lrt","-lgomp"], extra_compile_args=["-fopenmp", "-O3"])

if "--openmp" in sys.argv:
    sys.argv.remove("--openmp")
if "--noopenmp" in sys.argv:
    sys.argv.remove("--noopenmp")


setup (name = 'neighborsMinHash',
        version = '0.1',
        description = 'Builds inverse index for minHash algorithm.',
        ext_modules = [module1],
        packages = ['neighbors', 'computation'],
        package_dir = {'': 'lib'})
