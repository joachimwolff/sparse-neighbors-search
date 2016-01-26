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

# # call minHash without any optimization parameter
# ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
# --max_bin_size  50 --minimal_blocks_in_common 1 \
# --shingle_size 4 --excess_factor 5 \
# --similarity False --bloomierFilter False \
# --number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
# --prune_inverse_index_after_instance -1.0 \
# --removeHashFunctionWithLessEntriesAs -1 \
# --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/init_values
# # store it on disk
# # read file by python script to get the values and store them in the shell script

# # Best pruning value for the frequency of occurences of hash values
./getFrequencyOfOccurencesOfHashValues.py -i result/init_values -o result/max_value_init
MAX_VALUE_OCCURENCE_OF_HASH_VALUE=`cat result/max_value_init`
# echo $MAX_VALUE_OCCURENCE_OF_HASH_VALUE
# for (( i=0; i<=$MAX_VALUE_OCCURENCE_OF_HASH_VALUE; i++ ))
# do
#     echo $i
#             ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
# --max_bin_size  50 --minimal_blocks_in_common 1 \
# --shingle_size 4 --excess_factor 5 \
# --similarity False --bloomierFilter False \
# --number_of_cores 4 --chunk_size 0 --prune_inverse_index ${i} \
# --prune_inverse_index_after_instance -1.0 \
# --removeHashFunctionWithLessEntriesAs -1 \
# --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/prune_index_frequencey+${i}
#     echo "DONE"
# done 



# # Removing hash functions with less hash values than n
./getSizeOfHashFunctions.py -i result/init_values -o result/size_of_hash_functions
MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION=`cat result/size_of_hash_functions`
# echo $MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION
# for i in $MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION
# do
#     echo $i
#             ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
# --max_bin_size  50 --minimal_blocks_in_common 1 \
# --shingle_size 4 --excess_factor 5 \
# --similarity False --bloomierFilter False \
# --number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
# --prune_inverse_index_after_instance -1.0 \
# --removeHashFunctionWithLessEntriesAs ${i} \
# --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/size_of_hash_function_pruned+${i}
#     echo "DONE"
# done 
# # Influence of shingels
# for (( i=0; i<=10; i++ ))
# do
#     echo $i
#             ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
# --max_bin_size  50 --minimal_blocks_in_common 1 \
# --shingle_size ${i} --excess_factor 5 \
# --similarity False --bloomierFilter False \
# --number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
# --prune_inverse_index_after_instance -1.0 \
# --removeHashFunctionWithLessEntriesAs -1 \
# --hash_algorithm  0 --block_size  1 --shingle 1 --file_name result/shingle+${i}
#     echo "DONE"
# done 
# Frequencey of pruning

### Best pruning value for the frequency of occurences of hash values
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    for (( j=0; j<=$MAX_VALUE_OCCURENCE_OF_HASH_VALUE; j++ ))
    do
        echo $i
        echo $j
                    ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
        --max_bin_size  50 --minimal_blocks_in_common 1 \
        --shingle_size 4 --excess_factor 5 \
        --similarity False --bloomierFilter False \
        --number_of_cores 4 --chunk_size 0 --prune_inverse_index ${j} \
        --prune_inverse_index_after_instance ${i} \
        --removeHashFunctionWithLessEntriesAs -1 \
        --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/freqency_pruning_and_occurences+${i}+${j}
            echo "DONE"
    done
done
### Removing hash functions with less hash values than n
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    for j in $MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION
    do
        echo $i
        echo $j
                    ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
        --max_bin_size  50 --minimal_blocks_in_common 1 \
        --shingle_size 4 --excess_factor 5 \
        --similarity False --bloomierFilter False \
        --number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
        --prune_inverse_index_after_instance ${i} \
        --removeHashFunctionWithLessEntriesAs ${j} \
        --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/frequencey_pruning_and_size_hash_values+${i}+${j}
            echo "DONE"
    done
done


for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
    for (( j=0; j<=$MAX_VALUE_OCCURENCE_OF_HASH_VALUE; j++ ))
        do
        for k in $MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION
            do
                for (( l=0; l<=10; l++ ))
                    do
                    echo $i
                    echo $j
                                ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
                    --max_bin_size  50 --minimal_blocks_in_common 1 \
                    --shingle_size {l} --excess_factor 5 \
                    --similarity False --bloomierFilter False \
                    --number_of_cores 4 --chunk_size 0 --prune_inverse_index {k} \
                    --prune_inverse_index_after_instance ${i} \
                    --removeHashFunctionWithLessEntriesAs ${j} \
                    --hash_algorithm  0 --block_size  1 --shingle 1 --file_name result/all+${i}+${j}+${k}+${l}
                        echo "DONE"
                    done
            done
       done
done

