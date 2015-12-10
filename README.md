# Approximate nearest neighbors search on sparse datasets
========
With minHashNearestNeighbors it is possible to search the k-nearest neighbors 
inside a sparse data structure. 

To use it:

    from neighborsMinHash import MinHash
    minHash = MinHash()
    minHash.fit(X)
    minHash.kneighbors(return_distance=False)

Features
--------

- Efficient approximate k-nearest neighbors search
- works only on sparse datasets

Installation
------------

Install minHashNearestNeighbors by running:

    python setup.py install

In a user only context:

	python setup.py install --user

On MAC OS X the default compiler doesn't support openmp, it is deactivated by default. If you want to compile with openmp support, add the flag "--openmp":
	
	python setup.py install --user --openmp

On a Linux system openmp is default. If you don't want to use it set:
	
	python setup.py install --user --noopenmp

Instead of cloning the repository via git clone and than running the installation, you can do it in one step with pip:
	
	pip install git+https://github.com/joachimwolff/minHashNearestNeighbors.git



Uninstall
---------
To delete neighborsMinHash run the following command:

	pip uninstall neighborsMinHash

If you have run the uninstall command and want to make sure everything is gone, look at your python installation directory.
If you have used the --user flag the path in Ubuntu 14.04 is:

	~/.local/lib/python2.7/site-packages


Contribute
----------

- Source Code: https://github.com/joachimwolff/minHashNearestNeighbors

Support
-------

If you are having issues, please let me know.
Mail adress: wolffj@informatik.uni-freiburg.de

