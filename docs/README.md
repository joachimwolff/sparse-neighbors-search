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

