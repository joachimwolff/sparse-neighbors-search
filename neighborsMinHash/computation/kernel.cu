#include <stdio.h>
#include "sparseMatrix.h"
#include <math.h>
#define float A = sqrt(2) - 1;

__global__ void fit(SparseMatrix* pRawData, int pNumberOfHashFunctions, ) {
    __shared__ int signature[pNumberOfHashFunctions];
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int minHashValue = MAX_VALUE;
    int hashValue = 0;
    if (blockId < pRawData->size()) {
        int sizeOfInstance = pRawData->getSizeOfInstance(blockId);
        for (int i = 0; i < sizeOfInstance; ++i) {
            hashValue = computeHashValue(pRawData->getNextElement(blockId, i));
            if (hashValue < minHashValue) {
                minHashValue = hashValue;
            }
        }
        signature[threadId] = minHashValue;
    }
    __syncthread();
    
    if (threadId == 0) {
        // put signature to inverse index
    }
}

__device__ int computeHashValue(int key, int aModulo) {
          // source:  Thomas Wang: Integer Hash Functions, 1997 / 2007 
          // https://gist.github.com/badboy/6267743
          key = key * A;
          key = ~key + (key << 15);
          key = key ^ (key >> 12);
          key = key + (key << 2);
          key = key ^ (key >> 4);
          key = key * 2057;
          key = key ^ (key >> 16);
          return key % aModulo;
}