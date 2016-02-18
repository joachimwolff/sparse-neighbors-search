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
# Albert-Ludwigs-University Freiburg im Breisgau

## CUDA extension based on the example by Robert McGibbon and Yutong Zhao
## https://github.com/rmcgibbo/npcuda-example

import time 
__author__ = "Joachim Wolff"
__contact__ = "wolffj@informatik.uni-freiburg.de"
__copyright__ = "Copyright 2016, Joachim Wolff"
__credits__ = ["Milad Miladi", "Fabrizio Costa"]
__license__ = "No License"
__date__ = time.strftime("%d/%m/%Y")
__version_ = "0.1.dev"

from setuptools import setup, find_packages
# from distutils.core import Extension
import platform
import sys


import  os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy


sources_list = ['neighborsMinHash/computation/interface/minHash_PythonInterface.cpp', 'neighborsMinHash/computation/minHash.cpp', 
                'neighborsMinHash/computation/minHashBase.cpp', 'neighborsMinHash/computation/inverseIndex.cpp',
                 'neighborsMinHash/computation/inverseIndexStorageBloomierFilter.cpp' , 'neighborsMinHash/computation/inverseIndexStorageUnorderedMap.cpp',
                 'neighborsMinHash/computation/bloomierFilter/bloomierFilter.cpp', 'neighborsMinHash/computation/bloomierFilter/bloomierHash.cpp',
                 'neighborsMinHash/computation/bloomierFilter/encoder.cpp','neighborsMinHash/computation/bloomierFilter/orderAndMatchFinder.cpp']
depends_list = ['neighborsMinHash/computation/minHash.h', 'neighborsMinHash/computation/minHashBase.h', 'neighborsMinHash/computation/inverseIndex.h', 'neighborsMinHash/computation/kSizeSortedMap.h',
         'neighborsMinHash/computation/typeDefinitions.h', 'neighborsMinHash/computation/parsePythonToCpp.h', 'neighborsMinHash/computation/sparseMatrix.h',
          'neighborsMinHash/computation/inverseIndexStorage.h', 
                 'neighborsMinHash/computation/inverseIndexStorageBloomierFilter.h' , 'neighborsMinHash/computation/inverseIndexStorageUnorderedMap.h',
                 'neighborsMinHash/computation/bloomierFilter/bloomierFilter.h', 'neighborsMinHash/computation/bloomierFilter/bloomierHash.h',
                 'neighborsMinHash/computation/bloomierFilter/encoder.h','neighborsMinHash/computation/bloomierFilter/orderAndMatchFinder.h', 'neighborsMinHash/computation/hash.h']
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
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            # raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
            return None

    return cudaconfig
CUDA = locate_cuda()


# Obtain the numpy include directory.  This logic works across numpy versions.
# try:
#     numpy_include = numpy.get_include()
# except AttributeError:
#     numpy_include = numpy.get_numpy_include()






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
        if os.path.splitext(src)[1] == '.cu':
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

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



if (locate_cuda == None or no_cuda):
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
else:
    print "CUDA found on system. Installing MinHash with CUDA-Support."
    sources_list.extend(['neighborsMinHash/computation/kernel.cu', 'neighborsMinHash/computation/inverseIndexCuda.cu'])
    depends_list.extend(['neighborsMinHash/computation/kernel.h', 'neighborsMinHash/computation/inverseIndexCuda.h'])
    # Extension('_minHash', sources = sources_list, depends = depends_list,
    #      define_macros=[('OPENMP', None)], extra_link_args = ["-lm", "-lrt","-lgomp"], 
    #     extra_compile_args=["-fopenmp", "-O3", "-std=c++11"])
    ext = Extension('_minHash',
                sources = sources_list, depends = depends_list,
                library_dirs=[CUDA['lib64']],
                libraries=['cudart'],
                language='c++',
                runtime_library_dirs=[CUDA['lib64']],
                # this syntax is specific to this build system
                # we're only going to use certain compiler args with nvcc and not with gcc
                # the implementation of this trick is in customize_compiler() below
                define_macros=[('OPENMP', None), ('CUDA', None)],
                # extra_link_args={'gcc': ["-lm", "-lrt","-lgomp"], 
                #                   'nvcc' :[]  },
                extra_link_args=["-lm", "-lrt","-lgomp"],
                extra_compile_args={'gcc': ["-fopenmp", "-O3", "-std=c++11"],
                                    'nvcc': ['-arch=sm_20', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'", '-std=c++11' ]},
                include_dirs = [CUDA['include'], 'src', '/home/joachim/Software/cub-1.5.1'],
                platforms = "Linux, Mac OS X"
                )
                
    setup(name='neighborsMinHash',
        # random metadata. there's more you can supploy
        author='Joachim Wolff',
        ext_modules = [ext],
    
        # inject our custom trigger
        cmdclass={'build_ext': custom_build_ext},
    
        # since the package has c code, the egg cannot be zipped
        zip_safe=False,
        author_email = 'wolffj@informatik.uni-freiburg.de',
        url='https://github.com/joachimwolff/minHashNearestNeighbors',
        license='LICENSE',
        description='An approximate computation of nearest neighbors based on locality sensitive hash functions.',
        long_description=open('README.md').read(),
        install_requires=[
        "numpy >= 1.8.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.16.0",],
        # ext_modules = [module1],
        packages=['neighborsMinHash',
                    'neighborsMinHash.neighbors',
                    'neighborsMinHash.util',
                    'neighborsMinHash.clustering',
                    #  'neighborsMinHash.computation',
                ],
        platforms = "Linux, Mac OS X",
        version = '0.1.dev')