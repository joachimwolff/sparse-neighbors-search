# Copyright 2015 Joachim Wolff
# Master Thesis
# Tutors: Milad Miladi, Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau

__author__ = 'joachimwolff'

from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
import logging

import _minHashBloomierFilterClassifier