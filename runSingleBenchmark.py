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
from eden.converter.graph.gspan import gspan_to_eden
from eden.graph import Vectorizer
import time
import os.path
import cPickle as pickle
from sklearn.metrics import accuracy_score
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_neighbors',  type=int, dest="n_neighbors")
    parser.add_argument('-r', '--radius',type=float, dest="radius")
    parser.add_argument('-f', '--fast',type=bool, dest="fast")
    parser.add_argument('-nh', '--number_of_hash_functions',type=int, dest="number_of_hash_functions")    
    parser.add_argument('-m', '--max_bin_size', type=int,dest="max_bin_size")
    parser.add_argument('-mb', '--minimal_blocks_in_common', type=int,dest="minimal_blocks_in_common")
    parser.add_argument('-s', '--shingle_size', type=int,dest="shingle_size")
    parser.add_argument('-e', '--excess_factor',type=int, dest="excess_factor")
    parser.add_argument('-sim', '--similarity', type=bool,dest="similarity")
    parser.add_argument('-b', '--bloomierFilter', type=bool,dest="bloomierFilter")
    parser.add_argument('-nc', '--number_of_cores', type=int,dest="number_of_cores")
    parser.add_argument('-cs', '--chunk_size', type=int,dest="chunk_size")
    parser.add_argument('-p', '--prune_inverse_index',type=int, dest="prune_inverse_index")
    parser.add_argument('-pi', '--prune_inverse_index_after_instance',type=float, dest="prune_inverse_index_after_instance")
    parser.add_argument('-rh', '--removeHashFunctionWithLessEntriesAs',type=int, dest="removeHashFunctionWithLessEntriesAs")
    parser.add_argument('-ha', '--hash_algorithm',type=int, dest="hash_algorithm")
    parser.add_argument('-bs', '--block_size', type=int,dest="block_size")
    parser.add_argument('-sh', '--shingle',type=int, dest="shingle")
    parser.add_argument('-fn', '--file_name', type=str,dest="file_name")
    parser.add_argument('-lsb', '--least_significant_bit', type=str,dest="lsb")
    
    args = parser.parse_args()
    file_name = args.file_name
    # print args.n_neighbors
    # print args.radius
    # print args.fast
    # print args.number_of_hash_functions
    # print args.max_bin_size
    print "Pruning value: ", int(args.prune_inverse_index)
    print "File name: ", file_name
    
    if not os.path.isfile("bursiDataset"):
        graphs = gspan_to_eden( 'http://www.bioinf.uni-freiburg.de/~costa/bursi.gspan' )
        vectorizer = Vectorizer( r=2,d=5 )
        datasetBursi = vectorizer.transform( graphs )
        fileObject = open("bursiDataset",'wb')
        pickle.dump(datasetBursi,fileObject)
        fileObject.close()
    else:
        fileObject = open("bursiDataset",'r')
        datasetBursi = pickle.load(fileObject)
    fitting_time = 0
    query_time = 0
    minHash = MinHash(n_neighbors=int(args.n_neighbors), radius=float(args.radius), fast=bool(args.fast), 
                number_of_hash_functions=int(args.number_of_hash_functions),
                max_bin_size = int(args.max_bin_size), minimal_blocks_in_common = int(args.minimal_blocks_in_common),
                shingle_size = int(args.shingle_size), excess_factor = int(args.excess_factor),
                similarity=bool(args.similarity), bloomierFilter=False,
                number_of_cores=int(args.number_of_cores),
                chunk_size=int(args.chunk_size), prune_inverse_index=int(args.prune_inverse_index),
                prune_inverse_index_after_instance=float(args.prune_inverse_index_after_instance),
                removeHashFunctionWithLessEntriesAs=int(args.removeHashFunctionWithLessEntriesAs), 
                hash_algorithm = int(args.hash_algorithm),
                 block_size = int(args.block_size), 
                 shingle = int(args.shingle), remove_value_with_least_sigificant_bit=int(args.lsb))
                 
    time_start = time.time()
    minHash.fit(datasetBursi)
    
    fitting_time = time.time() - time_start
    distribution =  minHash.get_distribution_of_inverse_index()
    
    time_start = time.time()
    neighbors_result = minHash.kneighbors(fast=False,return_distance=False)
    
    query_time = time.time() - time_start
    if not os.path.isfile("neighbors_sklearn"):
        nearest_Neighbors = NearestNeighbors(n_jobs=4)
        nearest_Neighbors.fit(datasetBursi)
        neighbors_sklearn = nearest_Neighbors.kneighbors(return_distance=False)
        fileObject = open("neighbors_sklearn",'wb')
        pickle.dump(neighbors_sklearn,fileObject)
        fileObject.close()
    else:
        fileObject = open("neighbors_sklearn",'r')
        neighbors_sklearn = pickle.load(fileObject)
        
    accuracy_score_ = 0.0
    
    for x, y in zip(neighbors_result, neighbors_sklearn):
        accuracy_score_ += accuracy_score(x, y)
    accuracy_score_ = accuracy_score_ / float(len(neighbors_result))
    
    output_data = [distribution, fitting_time, query_time, accuracy_score_]
    fileObject = open(file_name,'wb')
    pickle.dump(output_data,fileObject)
    fileObject.close()