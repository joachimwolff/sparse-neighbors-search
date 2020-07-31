#! /usr/bin/python
# Copyright 2016, 2017, 2018, 2019, 2020 Joachim Wolff
# PhD Thesis
#
# Copyright 2015, 2016 Joachim Wolff
# Master Thesis
# Tutor: Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwigs-University Freiburg im Breisgau

## CUDA extension based on the example by Robert McGibbon and Yutong Zhao
## https://github.com/rmcgibbo/npcuda-example
# Copyright (c) 2014, Robert T. McGibbon and the Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
import time 
__author__ = "Joachim Wolff"
__contact__ = "wolffj@informatik.uni-freiburg.de"
__copyright__ = "Copyright 2020, Joachim Wolff"
__credits__ = ["Milad Miladi", "Fabrizio Costa"]
__license__ = "MIT"
__date__ = time.strftime("%d/%m/%Y")
__version__ = "0.5"

from setuptools import setup, find_packages
import platform
import sys


import  os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy

import distutils.sysconfig
import distutils.ccompiler

sources_list = ['sparse_neighbors_search/computation/interface/nearestNeighbors_PythonInterface.cpp', 'sparse_neighbors_search/computation/nearestNeighbors.cpp', 
                 'sparse_neighbors_search/computation/inverseIndex.cpp', 'sparse_neighbors_search/computation/inverseIndexStorageUnorderedMap.cpp']
depends_list = ['sparse_neighbors_search/computation/nearestNeighbors.h', 'sparse_neighbors_search/computation/inverseIndex.h', 'sparse_neighbors_search/computation/kSizeSortedMap.h',
         'sparse_neighbors_search/computation/typeDefinitions.h', 'sparse_neighbors_search/computation/parsePythonToCpp.h', 'sparse_neighbors_search/computation/sparseMatrix.h',
          'sparse_neighbors_search/computation/inverseIndexStorage.h', 'sparse_neighbors_search/computation/inverseIndexStorageUnorderedMap.h','sparse_neighbors_search/computation/sseExtension.h', 'sparse_neighbors_search/computation/avxExtension.h''sparse_neighbors_search/computation/hash.h']
openmp = True
if "--openmp" in sys.argv:
    module1 = Extension('_nearestNeighbors', sources = sources_list, depends = depends_list,
         define_macros=[('OPENMP', None)], extra_link_args = ["-lm", "-lrt","-lgomp"], 
        extra_compile_args=["-fopenmp", "-O3", "-std=c++11", "-funroll-loops", "-mavx"]) # , "-msse4.1"
elif platform.system() == 'Darwin' or "--noopenmp" in sys.argv:
    module1 = Extension('_nearestNeighbors', sources = sources_list, depends = depends_list, 
        extra_compile_args=["-O3", "-std=c++11", "-funroll-loops", "-mavx"])
    openmp = False

else:
    module1 = Extension('_nearestNeighbors', sources = sources_list, depends = depends_list,
        define_macros=[('OPENMP', None)], extra_link_args = ["-lm", "-lrt","-lgomp"],
         extra_compile_args=["-fopenmp", "-O3", "-std=c++11", "-funroll-loops", "-mavx"])
no_cuda = False

if "--nocuda" in sys.argv:
    no_cuda = True
    sys.argv.remove("--nocuda")
    
if "--openmp" in sys.argv:
    sys.argv.remove("--openmp")
if "--noopenmp" in sys.argv:
    sys.argv.remove("--noopenmp")



def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            print ('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(cudaconfig[k]):
            # raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
            return None

    return cudaconfig
CUDA = locate_cuda()


def customize_compiler_gcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile



def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu' or os.path.splitext(src)[1] == '.cuh':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so
        if os.path.splitext(src)[1] == '.cpp' or os.path.splitext(src)[1] == '.h':
            if '-O2' in self.compiler_so:
                self.compiler_so.remove('-O2')
            if '-g' in self.compiler_so:
                self.compiler_so.remove('-g')
            if '-DNDEBUG' in self.compiler_so:
                self.compiler_so.remove('-DNDEBUG')

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



if (locate_cuda() == None or no_cuda):
    print("No Cuda found or no cuda forced. Installation without GPU support.")
    setup (name = 'sparse_neighbors_search',
            author = 'Joachim Wolff',
            author_email = 'wolffj@informatik.uni-freiburg.de',
            url='https://github.com/joachimwolff/minHashNearestNeighbors',
            license='MIT',
            description='An approximate computation of nearest neighbors based on locality sensitive hash functions.',
            long_description=open('README.md').read(),
            install_requires=[
            "numpy >= 1.17.0",
            "scipy >= 1.3.0",
            "scikit-learn >= 0.21.0",],
            ext_modules = [module1],
            packages=['sparse_neighbors_search',
                        'sparse_neighbors_search.neighbors',
                        'sparse_neighbors_search.cluster',
                    ],
            platforms = "Linux",
            version = __version__
            )
else:
    print ("CUDA found on system. Installing MinHash with CUDA-Support.")
    sources_list.extend(['sparse_neighbors_search/computation/kernel.cu', 'sparse_neighbors_search/computation/inverseIndexCuda.cu', 'sparse_neighbors_search/computation/nearestNeighborsCuda.cu'])
    depends_list.extend(['sparse_neighbors_search/computation/typeDefinitionsCuda.h', 'sparse_neighbors_search/computation/kernel.h', 'sparse_neighbors_search/computation/inverseIndexCuda.h', 'sparse_neighbors_search/computation/nearestNeighborsCuda.h', ])
    if openmp:
        ext = Extension('_nearestNeighbors',
                    sources = sources_list, depends = depends_list,
                    library_dirs=[CUDA['lib64']],
                    libraries=['cudart'],
                    language='c++',
                    runtime_library_dirs=[CUDA['lib64']],
                    # this syntax is specific to this build system
                    # we're only going to use certain compiler args with nvcc and not with gcc
                    # the implementation of this trick is in customize_compiler() below
                    define_macros=[('OPENMP', None), ('CUDA', None)],
                    extra_link_args=["-lm", "-lrt","-lgomp"],
                    extra_compile_args={'gcc': ["-fopenmp", "-O3", "-std=c++11", "-funroll-loops", "-mavx"],
                                        'nvcc': ['-arch=sm_60', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", '-std=c++11' ]},
                    include_dirs = [CUDA['include'], 'src'],
                    platforms = "Linux, Mac OS X"
                    )
    else:
        ext = Extension('_nearestNeighbors',
                    sources = sources_list, depends = depends_list,
                    library_dirs=[CUDA['lib64']],
                    libraries=['cudart'],
                    language='c++',
                    runtime_library_dirs=[CUDA['lib64']],
                    # this syntax is specific to this build system
                    # we're only going to use certain compiler args with nvcc and not with gcc
                    # the implementation of this trick is in customize_compiler() below
                    define_macros=[('CUDA', None)],
                    extra_link_args=["-lm", "-lrt","-lgomp"],
                    extra_compile_args={'gcc': ["-O3", "-std=c++11", "-mavx"],
                                        'nvcc': ['-arch=sm_60', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", '-std=c++11' ]},
                    include_dirs = [CUDA['include'], 'src'],
                    platforms = "Linux, Mac OS X"
                    )
                
    setup(name='sparse_neighbors_search',
        author='Joachim Wolff',
        ext_modules = [ext],
    
        # inject our custom trigger
        cmdclass={'build_ext': custom_build_ext},
    
        # since the package has c code, the egg cannot be zipped
        zip_safe=False,
        author_email = 'wolffj@informatik.uni-freiburg.de',
        url='https://github.com/joachimwolff/minHashNearestNeighbors',
        license='MIT',
        description='An approximate computation of nearest neighbors based on locality sensitive hash functions.',
        long_description=open('README.md').read(),
        install_requires=[
        "numpy >= 1.17.0",
        "scipy >= 1.3.0",
        "scikit-learn >= 0.21.0",],
        packages=['sparse_neighbors_search',
                    'sparse_neighbors_search.neighbors',
                    'sparse_neighbors_search.cluster',
                ],
        platforms = "Linux, Mac OS X",
        version = __version__)