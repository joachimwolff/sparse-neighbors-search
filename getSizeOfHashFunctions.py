#! /usr/bin/python
# Copyright 2016 Joachim Wolff
# Master Thesis
# Tutors: Milad Miladi, Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau
#
__author__ = 'wolffjoachim'

from neighborsMinHash import MinHash
from sklearn.neighbors import NearestNeighbors

import time
import os.path
import cPickle as pickle
from sklearn.metrics import accuracy_score
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
   
    parser.add_argument('-i', '--input_file_name', type=str,dest="input_file_name")
    parser.add_argument('-o', '--output_file_name', type=str,dest="output_file_name")
    parser.add_argument('-f', '--frequency', type=int,dest="frequency")
    
    args = parser.parse_args()
    input_file_name = args.input_file_name
    output_file_name = args.output_file_name
    
    
    if os.path.isfile(input_file_name):
        fileObject = open(input_file_name,'r')
        data = pickle.load(fileObject)
       
        fileObjectOut = open(output_file_name,'wb')
        
        for i in sorted(set(sorted(data[0][1][0::10]))):
            fileObjectOut.write(str(i)+" ")
        
                
        fileObjectOut.close()
       