/**
 Copyright 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa, Milad Miladi
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/

#include "nearestNeighborsCuda.h"
#include "kernel.h"
NearestNeighborsCuda::NearestNeighborsCuda() {
    
}
// NearestNeighborsCuda::NearestNeighborsCuda(int* pFeatureList, float* pValuesList,
//                                         int* pSizeOfInstanceList) {
//    mDev_FeatureList = pFeatureList;
//    mDev_ValuesList = pValuesList;
//    mDev_SizeOfInstanceList = pSizeOfInstanceList;
// //    mMaxNnz = pMaxNnz;
// }
NearestNeighborsCuda::~NearestNeighborsCuda() {
    
}

cudaInstance* NearestNeighborsCuda::computeNearestNeighbors(neighborhood* neighbors, size_t pSimilarity, SparseMatrixFloat* pRawData,
                                                                int* pJumpLength) {
    
    
    // int* candidates
    // int* number of candidates per instance
    int* numberOfCandidatesPerInstance = (int*) malloc(neighbors->neighbors->size() * sizeof(int));
    // int* jump length within candidates
    // pJumpLength = (int*) malloc(neighbors->neighbors->size() * sizeof(int));
    int jumpLengthCount = 0;
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        numberOfCandidatesPerInstance[i] = static_cast<int>(neighbors->neighbors->operator[](i).size());
        pJumpLength[i] = jumpLengthCount;
        jumpLengthCount += static_cast<int>(neighbors->neighbors->operator[](i).size());
    }
    int* numberOfCandidatesPerInstanceCuda;
    cudaMalloc((void **) &numberOfCandidatesPerInstanceCuda, neighbors->neighbors->size() * sizeof(int));
    cudaMemcpy(numberOfCandidatesPerInstanceCuda, numberOfCandidatesPerInstance, neighbors->neighbors->size() * sizeof(int), 
                cudaMemcpyHostToDevice);
    int* jumpLengthCuda;
    cudaMalloc((void **) &jumpLengthCuda, neighbors->neighbors->size() * sizeof(int));
    cudaMemcpy(jumpLengthCuda, pJumpLength, neighbors->neighbors->size() * sizeof(int), 
                cudaMemcpyHostToDevice);
    cudaInstance* candidates = (cudaInstance*) malloc(jumpLengthCount * sizeof(cudaInstance));
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            candidates[pJumpLength[i] + j].x = neighbors->neighbors->operator[](i)[j];
        }
    }
    
    cudaInstance* candidatesCuda;
    cudaMalloc((void **) &candidatesCuda, jumpLengthCount * sizeof(cudaInstance));
    cudaMemcpy(candidatesCuda, candidates, jumpLengthCount * sizeof(cudaInstance), 
                cudaMemcpyHostToDevice);
  
    if (pSimilarity) {
        // cosineSimilarityCuda<<<32, 96>>>(d_data, size, sizeOfCandidatesCuda, mDev_FeatureList,
        //                                 mDev_ValuesList, mDev_SizeOfInstanceList, mMaxNnz);
    } else {
        euclideanDistanceCuda<<<128, 128>>>(candidatesCuda, jumpLengthCuda, numberOfCandidatesPerInstanceCuda, 
                                        neighbors->neighbors->size(), (*mDev_FeatureList),
                                        (*mDev_ValuesList), (*mDev_SizeOfInstanceList), 
                                        (*mDev_JumpLengthList), (*mDev_DotProducts));
        cudaDeviceSynchronize();
    }
    cudaMemcpy(candidates, candidatesCuda, sizeof(cudaInstance) * jumpLengthCount, cudaMemcpyDeviceToHost);
    cudaFree(candidatesCuda);
    free(numberOfCandidatesPerInstance);
    // free(jumpLength);
    cudaFree(numberOfCandidatesPerInstanceCuda);
    cudaFree(jumpLengthCuda);
    return candidates;
}