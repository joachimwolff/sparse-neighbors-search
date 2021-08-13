# Approximate k-nearest neighbors search on sparse datasets
With MinHash and WTA-Hash it is possible to search the approximate k-nearest neighbors 
within a sparse data structure. It works best for very high dimensional and very sparse datasets, e.g. one million dimensions and 400 non-zero feature ids on average.

To use it:

    from sparse_neighbors_search import MinHash
    minHash = MinHash()
    minHash.fit(X)
    minHash.kneighbors(return_distance=False)


Citation
--------

Joachim Wolff, Rolf Backofen, Björn Grüning. **Robust and efficient single-cell Hi-C clustering with approximate k-nearest neighbor graphs, Bioinformatics**, btab394, https://doi.org/10.1093/bioinformatics/btab394


Disclaimer
----------

With the update to version 0.3 'Charlie Brown' which needs support for SSE4.1 by your operating system and cpu macOS is no longer supported. Feel free to use it and or to get it run on this platform but I cannot test it there and probably it will not run.

Version 0.4 'Lucy van Pelt' drops Python 2 support and introduces the support of Python 3. Additional, CUDA compile level is set to sm_60, requiring CUDA 8 and a 'Pascal'  GPU architecture, GTX 10X0 family. However, nothing in the source code is changed feel free to set it to an older version.

Version 0.6 'Linus van Pelt' removes official support for CUDA. There is a compiling issue and I am unable to resolve it. Maybe it will comeback in the future.

Features
--------

- Efficient approximate k-nearest neighbors search
- works only on sparse datasets

## Installation

#### Installation via conda

The package is available via the bioconda channel:

	conda install sparse-neighbors-search -c bioconda


#### Installation from source

Install sparse_neighbors_search by running:

    python setup.py install

In a user only context:

	python setup.py install --user

On MAC OS X the default compiler doesn't support openmp, it is deactivated by default. If you want to compile with openmp support, add the flag "--openmp":
	
	python setup.py install --user --openmp

On a Linux system openmp is default. If you don't want to use it set:
	
	python setup.py install --user --noopenmp

Version 0.6 drops the support of CUDA. However, if you want to try to compile it on your **own risk**:
	
	python setup.py install --cuda
	
Instead of cloning the repository via git clone and than running the installation, you can do it in one step with pip:
	
	pip install git+https://github.com/joachimwolff/minHashNearestNeighbors.git

The installation requires g++ and the C++11 libs, numpy, scikit-learn, cython, scipy, openmp and SSE 4.1 support.

The software was tested on Ubuntu 20.04 with g++ 7.5, numpy 1.19, scikit-learn 0.23, Cython 0.29 and scipy 1.5.


Contribute
----------

- Source Code: https://github.com/joachimwolff/minHashNearestNeighbors

Support
-------

If you are having issues, please let me know.
Mail address: wolffj[at]informatik[dot]uni-freiburg[dot]de

