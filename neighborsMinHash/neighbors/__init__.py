# Copyright 2015 Joachim Wolff
# Master Project
# Tutors: Milad Miladi, Fabrizio Costa
# Summer semester 2015
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau

__author__ = 'joachimwolff'

# import dill

# def run_dill_encoded(what):
#     """Use dill as replacement for pickle to enable multiprocessing on instance methods"""
#     fun, args = dill.loads(what)
#     return fun(*args)
# def apply_async(pool, fun, args, callback=None):
#     """
#     Wrapper around apply_async() from multiprocessing, to use dill instead of pickle.
#     This is a workaround to enable multiprocessing of classes.
#     """
#     # return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),), callback=callback)
#     return pool.apply(run_dill_encoded, (dill.dumps((fun, args)),))

from minHashNearestNeighbors import MinHashNearestNeighbors
from minHashKNeighborsClassifier import MinHashKNeighborsClassifier

# results = [pool.apply(cube, args=(x,)) for x in range(1,7)]