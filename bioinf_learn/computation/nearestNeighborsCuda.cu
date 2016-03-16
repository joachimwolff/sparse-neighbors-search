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
NearestNeighborsCuda::NearestNeighborsCuda(size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz) {
   mDev_FeatureList = pFeatureList;
   mDev_ValuesList = pValuesList;
   mDev_SizeOfInstanceList = pSizeOfInstanceList;
   mMaxNnz = pMaxNnz;
}
NearestNeighborsCuda::~NearestNeighborsCuda() {
    
}

cudaInstanceVector* NearestNeighborsCuda::computeNearestNeighbors(neighborhood* neighbors, size_t pSimilarity) {
    cudaInstanceVector* candidates = (cudaInstanceVector*) malloc(sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    uint* sizeOfCandidates = (uint*) malloc (sizeof(uint) * neighbors->neighbors->size());
    
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        candidates[i].instance = (cudaInstance*) malloc(sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size());
        sizeOfCandidates[i] = neighbors->neighbors->operator[](i).size();
        // printf ("size: %i", sizeOfCandidates[i]);
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            candidates[i].instance[j].x = neighbors->neighbors->operator[](i)[j];
            candidates[i].instance[j].y = neighbors->distances->operator[](i)[j];
        }        
    }
    
    cudaInstanceVector* candidatesCuda;
    cudaInstanceVector* candidatesCudaHost = (cudaInstanceVector*) malloc(sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        cudaMalloc(&(candidatesCudaHost[i].instance), sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size());
        cudaMemcpy(candidatesCudaHost[i].instance, candidates[i].instance, sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size(), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&candidatesCuda, sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    cudaMemcpy(candidatesCuda, candidatesCudaHost, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyHostToDevice);
    
    uint* sizeOfCandidatesCuda;
    uint size = neighbors->neighbors->size();
    cudaMalloc(&sizeOfCandidatesCuda, sizeof(uint) * neighbors->neighbors->size());
    cudaMemcpy(sizeOfCandidatesCuda, sizeOfCandidates, sizeof(uint) * neighbors->neighbors->size(), cudaMemcpyHostToDevice);
    if (pSimilarity) {
        cosineSimilarityCuda<<<128, 128>>>(candidatesCuda, size, sizeOfCandidatesCuda, mDev_FeatureList,
                                        mDev_ValuesList, mDev_SizeOfInstanceList, mMaxNnz);
    } else {
        euclideanDistanceCuda<<<128, 128>>>(candidatesCuda, size, sizeOfCandidatesCuda, mDev_FeatureList,
                                        mDev_ValuesList, mDev_SizeOfInstanceList, mMaxNnz);
    }
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        cudaMemcpy(candidates[i].instance, candidatesCudaHost[i].instance, sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size(), cudaMemcpyDeviceToHost);
        cudaFree(candidatesCudaHost[i].instance);
    }
    cudaMemcpy(candidatesCudaHost, candidates, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyDeviceToHost);
    cudaFree(candidatesCudaHost);
    cudaFree(sizeOfCandidatesCuda);
    free(candidatesCudaHost);
    free(sizeOfCandidates);
    return candidates;
}