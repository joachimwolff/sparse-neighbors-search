# Approximate k-nearest neighbors search on sparse datasets
With MinHash and WTA-Hash from the bioinf-learn package it is possible to search the approximate k-nearest neighbors 
within a sparse data structure. It works best for very high dimensional and very sparse datasets, e.g. one million dimensions and 400 non-zero feature ids on average.

To use it:

    from bioinf_learn import MinHash
    minHash = MinHash()
    minHash.fit(X)
    minHash.kneighbors(return_distance=False)

Features
--------

- Efficient approximate k-nearest neighbors search
- works only on sparse datasets

Installation
------------

Install bioinf-learn by running:

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
To delete bioinf-learn run the following command:

	pip uninstall bioinf-learn

If you have run the uninstall command and want to make sure everything is gone, look at your python installation directory.
If you have used the --user flag the path in Ubuntu 14.04 is:

	~/.local/lib/python2.7/site-packages


Contribute
----------

- Source Code: https://github.com/joachimwolff/minHashNearestNeighbors

Support
-------

If you are having issues, please let me know.
Mail adress: wolffj[at]informatik[dot]uni-freiburg[dot]de

