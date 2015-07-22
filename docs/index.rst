.. minHashNearestNeighbors documentation master file, created by
   sphinx-quickstart on Wed Jul 22 13:43:03 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to minHashNearestNeighbors's documentation!
===================================================

Contents:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

# minHashNearestNeighbors
========
With minHashNearestNeighbors it is possible to search the k-nearest neighbors 
inside a sparse data structure. 

To use it:

    from neighbors import MinHashNearestNeighbors
    minHash = MinHashNearestNeighbors()
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

Contribute
----------

- Source Code: https://github.com/joachimwolff/minHashNearestNeighbors

Support
-------

If you are having issues, please let us know.
Mail adress: wolffj@informatik.uni-freiburg.de
