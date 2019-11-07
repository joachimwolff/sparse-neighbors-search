# Approximate k-nearest neighbors search on sparse datasets
With MinHash and WTA-Hash from the bioinf-learn package it is possible to search the approximate k-nearest neighbors 
within a sparse data structure. It works best for very high dimensional and very sparse datasets, e.g. one million dimensions and 400 non-zero feature ids on average.

To use it:

    from sparse_neighbors_search import MinHash
    minHash = MinHash()
    minHash.fit(X)
    minHash.kneighbors(return_distance=False)

Disclaimer
----------

With the update to version 0.3 'Charlie Brown' which needs support for SSE4.1 by your operating system and cpu macOS is no longer supported. Feel free to use it and or to get it run on this platform but I cannot test it there and probably it will not run.

Version 0.4 'Lucy van Pelt' drops Python 2 support and introduces the support of Python 3. Additional, CUDA compile level is set to sm_60, requiring CUDA 8 and a 'Pascal'  GPU architecture, GTX 10X0 family. However, nothing in the source code is changed feel free to set it to an older version.

Features
--------

- Efficient approximate k-nearest neighbors search
- works only on sparse datasets

Installation
------------

Install sparse_neighbors_search by running:

    python setup.py install

In a user only context:

	python setup.py install --user

On MAC OS X the default compiler doesn't support openmp, it is deactivated by default. If you want to compile with openmp support, add the flag "--openmp":
	
	python setup.py install --user --openmp

On a Linux system openmp is default. If you don't want to use it set:
	
	python setup.py install --user --noopenmp

GPU support is provided with Nvidias CUDA. If the setup detects a CUDA installation it is using it. If you want to force an installation without CUDA add the parameter:
	--nocuda

Instead of cloning the repository via git clone and than running the installation, you can do it in one step with pip:
	
	pip install git+https://github.com/joachimwolff/minHashNearestNeighbors.git

The installation requires g++ and the C++11 libs, numpy, scikit-learn, cython and scipy. For the GPU support the CUDA framework with nvcc.
The software was tested on Ubuntu 14.04 with g++ 4.8, CUDA 7.5, numpy 1.10.1, scikit-learn 0.17, Cython 0.23.4 and scipy 0.16.1.

Uninstall
---------
To delete sparse-neighbors-search run the following command:

	pip uninstall sparse-neighbors-search

If you have run the uninstall command and want to make sure everything is gone, look at your python installation directory.
If you have used the --user flag the path in Ubuntu 14.04 is:

	~/.local/lib/python2.7/site-packages


Contribute
----------

- Source Code: https://github.com/joachimwolff/minHashNearestNeighbors

Support
-------

If you are having issues, please let me know.
Mail address: wolffj[at]informatik[dot]uni-freiburg[dot]de

