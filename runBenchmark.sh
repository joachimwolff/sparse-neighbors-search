#!/bin/bash

# shingle_size = 4,
# excess_factor = 5,
# similarity=False, 
# bloomierFilter=False,
# number_of_cores=None, 
# chunk_size=None, 

# prune_inverse_index_after_instance=-1.0, 
# removeHashFunctionWithLessEntriesAs=-1, 
# hash_algorithm = 0, 
# block_size = 5, 
# shingle=0

# call minHash without any optimization parameter
./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
--max_bin_size  50 --minimal_blocks_in_common 1 \
--shingle_size 4 --excess_factor 5 \
--similarity False --bloomierFilter False \
--number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
--prune_inverse_index_after_instance -1.0 \
--removeHashFunctionWithLessEntriesAs -1 \
--hash_algorithm  0 --block_size  5 --shingle 0 --file_name init_values
# store it on disk
# read file by python script to get the values and store them in the shell script

./getFrequencyOfOccurencesOfHashValues.py -i init_values -o max_value_init
MAX_VALUE_OCCURENCE_OF_HASH_VALUE=`cat max_value_init`
echo $MAX_VALUE_OCCURENCE_OF_HASH_VALUE
for (( i=0; i<=$MAX_VALUE_OCCURENCE_OF_HASH_VALUE; i++ ))
do
    echo $i
            ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
--max_bin_size  50 --minimal_blocks_in_common 1 \
--shingle_size 4 --excess_factor 5 \
--similarity False --bloomierFilter False \
--number_of_cores 4 --chunk_size 0 --prune_inverse_index ${i} \
--prune_inverse_index_after_instance -1.0 \
--removeHashFunctionWithLessEntriesAs -1 \
--hash_algorithm  0 --block_size  5 --shingle 0 --file_name prune_index_frequencey+${i}
    echo "DONE"
done 



# Best pruning value for the frequency of occurences of hash values

# Removing hash functions with less hash values than n

# Influence of shingels

# Frequencey of pruning

### Best pruning value for the frequency of occurences of hash values

### Removing hash functions with less hash values than n
