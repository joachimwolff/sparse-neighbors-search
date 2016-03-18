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
NearestNeighborsCuda::NearestNeighborsCuda(int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int pMaxNnz) {
   mDev_FeatureList = pFeatureList;
   mDev_ValuesList = pValuesList;
   mDev_SizeOfInstanceList = pSizeOfInstanceList;
   mMaxNnz = pMaxNnz;
}
NearestNeighborsCuda::~NearestNeighborsCuda() {
    
}

cudaInstanceVector* NearestNeighborsCuda::computeNearestNeighbors(neighborhood* neighbors, size_t pSimilarity) {
    printf("30");
    fflush(stdout);
    cudaInstanceVector* candidates = (cudaInstanceVector*) malloc(sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    int* sizeOfCandidates = (int*) malloc (sizeof(int) * neighbors->neighbors->size());
    printf("33");
    fflush(stdout);
    
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        candidates[i].instance = (cudaInstance*) malloc(sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size());
        sizeOfCandidates[i] = neighbors->neighbors->operator[](i).size();
        if (i % 100 == 0) {
        printf ("CPUcandidate: %i, size: %i\n", i, sizeOfCandidates[i]);
        }
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            candidates[i].instance[j].x = static_cast<int>(neighbors->neighbors->operator[](i)[j]);
            //  if (i % 100 == 0)
            // printf("cpuinstanceid: %i ,size: %i ", candidates[i].instance[j].x, sizeOfCandidates[i]);
            // candidates[i].instance[j].y = neighbors->distances->operator[](i)[j];
        }  
        //  if (i % 100 == 0)
        // printf("\n");      
    }
    printf("44");
    fflush(stdout);
    
    // printf("size of neighbirs: %i",neighbors->neighbors->size());
    cudaInstanceVector* h_data = (cudaInstanceVector*) malloc(sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    memcpy(h_data, candidates, sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    // cudaInstanceVector* candidatesCudaHost = (cudaInstanceVector*) malloc(sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        // if (i == 130) {
        // printf("size instance: %i",  neighbors->neighbors->operator[](i).size());
        // }
        cudaMalloc(&(h_data[i].instance), sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size());
        cudaMemcpy(h_data[i].instance, candidates[i].instance, sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size(), cudaMemcpyHostToDevice);
    }
    printf("57");
    fflush(stdout);
    
    cudaInstanceVector* d_data;
    cudaMalloc(&d_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    cudaMemcpy(d_data, h_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyHostToDevice);
    printf("62");
    fflush(stdout);
    
    int* sizeOfCandidatesCuda;
    int size = neighbors->neighbors->size();
    cudaMalloc(&sizeOfCandidatesCuda, sizeof(int) * neighbors->neighbors->size());
    cudaMemcpy(sizeOfCandidatesCuda, sizeOfCandidates, sizeof(int) * neighbors->neighbors->size(), cudaMemcpyHostToDevice);
    if (pSimilarity) {
        cosineSimilarityCuda<<<32, 96>>>(d_data, size, sizeOfCandidatesCuda, mDev_FeatureList,
                                        mDev_ValuesList, mDev_SizeOfInstanceList, mMaxNnz);
    } else {
        euclideanDistanceCuda<<<32, 96>>>(d_data, size, sizeOfCandidatesCuda, mDev_FeatureList,
                                        mDev_ValuesList, mDev_SizeOfInstanceList, mMaxNnz);
    }
    printf("75");
    fflush(stdout);
    
    // cudaMemcpy(h_data, d_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyDeviceToHost);
    printf("78");
    fflush(stdout);
    
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        cudaMemcpy(candidates[i].instance, h_data[i].instance, sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size(), cudaMemcpyDeviceToHost);
        cudaFree(h_data[i].instance);
    }
    printf("84");
    fflush(stdout);
    
    // memcpy(candidates, h_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    // cudaMemcpy(candidates, d_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyDeviceToHost);
    printf("88");
    fflush(stdout);
    
    cudaFree(d_data);
    cudaFree(sizeOfCandidatesCuda);
    free(h_data);
    free(sizeOfCandidates);
    printf("94");
    fflush(stdout);
    
    return candidates;
}