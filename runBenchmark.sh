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
./runSingleBenchmark.py --n_neighbors 10 --radius 1.0 --fast False --number_of_hash_functions 400 \
--max_bin_size  50 --minimal_blocks_in_common 1 \
--shingle_size 4 --excess_factor 5 \
--similarity False --bloomierFilter False \
--number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
--prune_inverse_index_after_instance -1.0 \
--removeHashFunctionWithLessEntriesAs -1 \
--hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/init_values_wta -lsb 0
# store it on disk
# read file by python script to get the values and store them in the shell script

# Best pruning value for the frequency of occurences of hash values
# ./getFrequencyOfOccurencesOfHashValues.py -i result/init_values -o result/max_value_init
# MAX_VALUE_OCCURENCE_OF_HASH_VALUE=`cat result/max_value_init`
# echo $MAX_VALUE_OCCURENCE_OF_HASH_VALUE
# # MAX_VALUE_OCCURENCE_OF_HASH_VALUE=10
# for (( i=1; i<=$MAX_VALUE_OCCURENCE_OF_HASH_VALUE; i++ ))
# do
#     echo $i
#             ./runSingleBenchmark.py --n_neighbors 10 --radius 1.0 --fast False --number_of_hash_functions 400 \
# --max_bin_size  50 --minimal_blocks_in_common 1 \
# --shingle_size 4 --excess_factor 5 \
# --similarity False --bloomierFilter False \
# --number_of_cores 4 --chunk_size 0 --prune_inverse_index ${i} \
# --prune_inverse_index_after_instance -1.0 \
# --removeHashFunctionWithLessEntriesAs -1 \
# --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/prune_index_frequencey+${i} -lsb 0
#     echo "DONE"
# done 



# Removing hash functions with less hash values than ncc
./getSizeOfHashFunctions.py -i result/init_values_wta -o result/size_of_hash_functions_wta
MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION=`cat result/size_of_hash_functions`
echo $MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION
# for i in $MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION
# do
#     echo $i
#             ./runSingleBenchmark.py --n_neighbors 10 --radius 1.0 --fast False --number_of_hash_functions 400 \
# --max_bin_size  50 --minimal_blocks_in_common 1 \
# --shingle_size 4 --excess_factor 5 \
# --similarity False --bloomierFilter False \
# --number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
# --prune_inverse_index_after_instance -1.0 \
# --removeHashFunctionWithLessEntriesAs ${i} \
# --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/size_of_hash_function_pruned+${i} -lsb 0
#     echo "DONE"
# done 
# #Influence of shingels: concaternation
# for (( i=1; i<=10; i++ ))
# do
#     echo $i
#             ./runSingleBenchmark.py --n_neighbors 10 --radius 1.0 --fast False --number_of_hash_functions 400 \
# --max_bin_size  50 --minimal_blocks_in_common 1 \
# --shingle_size ${i} --excess_factor 5 \
# --similarity False --bloomierFilter False \
# --number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
# --prune_inverse_index_after_instance -1.0 \
# --removeHashFunctionWithLessEntriesAs -1 \
# --hash_algorithm  0 --block_size  1 --shingle 1 --file_name result/shingle_concat+${i} -lsb 0
#     echo "DONE"
# done 

# Influence of shingels: minimum value
# for (( i=1; i<=10; i++ ))
# do
#     echo $i
#             ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
# --max_bin_size  50 --minimal_blocks_in_common 1 \
# --shingle_size ${i} --excess_factor 5 \
# --similarity False --bloomierFilter False \
# --number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
# --prune_inverse_index_after_instance -1.0 \
# --removeHashFunctionWithLessEntriesAs -1 \
# --hash_algorithm  0 --block_size  1 --shingle 2 --file_name result/shingle_min_val+${i} -lsb 0
#     echo "DONE"
# done 
# Frequencey of pruning

### Best pruning value for the frequency of occurences of hash values
# for i in 0.1 0.5
# do
#     for (( j=1; j<=$MAX_VALUE_OCCURENCE_OF_HASH_VALUE; j++ ))
#     do
#         echo $i
#         echo $j
#                     ./runSingleBenchmark.py --n_neighbors 10 --radius 1.0 --fast False --number_of_hash_functions 400 \
#         --max_bin_size  50 --minimal_blocks_in_common 1 \
#         --shingle_size 4 --excess_factor 5 \
#         --similarity False --bloomierFilter False \
#         --number_of_cores 4 --chunk_size 0 --prune_inverse_index ${j} \
#         --prune_inverse_index_after_instance ${i} \
#         --removeHashFunctionWithLessEntriesAs -1 \
#         --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/freqency_pruning_and_occurences+${i}+${j} -lsb 0
#             echo "DONE"
#     done
# done


### Removing hash functions with less hash values than n
# ./getSizeOfHashFunctions.py -i result/init_values -o result/size_of_hash_functions -f 10
# MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION=`cat result/size_of_hash_functions`
# for i in 0.1 0.3 0.5 0.7
# do
#     for j in $MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION
#     do
#         echo $i
#         echo $j
#                     ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
#         --max_bin_size  50 --minimal_blocks_in_common 1 \
#         --shingle_size 4 --excess_factor 5 \
#         --similarity False --bloomierFilter False \
#         --number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
#         --prune_inverse_index_after_instance ${i} \
#         --removeHashFunctionWithLessEntriesAs ${j} \
#         --hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/frequencey_pruning_and_size_hash_values+${i}+${j} -lsb 0
#             echo "DONE"
#     done
# done

# remove least significant bit
for (( i=1; i<=10; i++ ))
do
    echo $i
            ./runSingleBenchmark.py --n_neighbors 10 --radius 1.0 --fast False --number_of_hash_functions 400 \
--max_bin_size  50 --minimal_blocks_in_common 1 \
--shingle_size 1 --excess_factor 5 \
--similarity False --bloomierFilter False \
--number_of_cores 4 --chunk_size 0 --prune_inverse_index -1 \
--prune_inverse_index_after_instance -1.0 \
--removeHashFunctionWithLessEntriesAs -1 \
--hash_algorithm  0 --block_size  1 --shingle 0 --file_name result/remove_least_significant_bit+${i} -lsb ${i}
    echo "DONE"
done 

# MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION="-1 "+$MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION
# for i in 0 1 # least siginificant bit
# do
#     for j in -1.0 0.5 # prune after instance
#     do
#         for k in -1 1 2 3 4 5 6 7 8 9 10 # prune occurences of hash values
#         do
#             for l in $MAX_VALUE_OCCURENCE_OF_HASH_FUNCTION # prune by size of hash function
#             do
#                 for (( m=1; m<=5; m++ )) # shingle concaternate
#                 do
#                     echo "lsb: " +${i} +${j}+${k}+${l}+${m}
#                     ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
#                     --max_bin_size  50 --minimal_blocks_in_common 1 \
#                     --shingle_size ${m} --excess_factor 5 \
#                     --similarity False --bloomierFilter False \
#                     --number_of_cores 4 --chunk_size 0 --prune_inverse_index ${k} \
#                     --prune_inverse_index_after_instance ${j} \
#                     --removeHashFunctionWithLessEntriesAs ${l} \
#                     --hash_algorithm  0 --block_size  1 --shingle 1 -lsb ${i} --file_name result/all+${i}+${j}+${k}+${l}+${m}
                    
#                 done
#                 for (( m=1; m<=3; m++ )) # shingle concaternate
#                 do
#                     echo "lsb: " +${i} +${j}+${k}+${l}+${m}
#                     ./runSingleBenchmark.py --n_neighbors 5 --radius 1.0 --fast False --number_of_hash_functions 400 \
#                     --max_bin_size  50 --minimal_blocks_in_common 1 \
#                     --shingle_size ${m} --excess_factor 5 \
#                     --similarity False --bloomierFilter False \
#                     --number_of_cores 4 --chunk_size 0 --prune_inverse_index ${k} \
#                     --prune_inverse_index_after_instance ${j} \
#                     --removeHashFunctionWithLessEntriesAs ${l} \
#                     --hash_algorithm  0 --block_size  1 --shingle 2 -lsb ${i} --file_name result/all+${i}+${j}+${k}+${l}+${m}
                    
#                 done
#             done
#         done
#     done
# done

