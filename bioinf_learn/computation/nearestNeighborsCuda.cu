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

cudaInstanceVector* NearestNeighborsCuda::computeNearestNeighbors(neighborhood* neighbors, size_t pSimilarity, SparseMatrixFloat* pRawData) {
    
    
    // int* candidates
    // int* number of candidates per instance
    int* numberOfCandidatesPerInstance = (int*) malloc(neighbors->neighbors->size() * sizeof(int));
    // int* jump length within candidates
    int* jumpLength = (int*) malloc(neighbors->neighbors->size() * sizeof(int));
    int jumpLengthCount = 0;
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        numberOfCandidatesPerInstance[i] = static_cast<int>(neighbors->neighbors->operator[](i).size());
        jumpLength[i] = jumpLengthCount;
        jumpLengthCount = static_cast<int>(neighbors->neighbors->operator[](i).size());
    }
    int* numberOfCandidatesPerInstanceCuda;
    cudaMalloc((void **) &numberOfCandidatesPerInstanceCuda, neighbors->neighbors->size() * sizeof(int));
    cudaMemcpy(numberOfCandidatesPerInstanceCuda, numberOfCandidatesPerInstance, neighbors->neighbors->size() * sizeof(int), 
                cudaMemcpyHostToDevice);
    int* jumpLengthCuda;
    cudaMalloc((void **) &jumpLengthCuda, neighbors->neighbors->size() * sizeof(int));
    cudaMemcpy(jumpLengthCuda, jumpLength, neighbors->neighbors->size() * sizeof(int), 
                cudaMemcpyHostToDevice);
    cudaInstance* candidates = (cudaInstance*) malloc(jumpLengthCount * sizeof(cudaInstance));
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            candidates[jumpLength[i] + j].x = neighbors->neighbors->operator[](i)[j];
        }
    }
    
    cudaInstance* candidatesCuda;
    cudaMalloc((void **) &candidatesCuda, jumpLengthCount * sizeof(cudaInstance));
    cudaMemcpy(candidatesCuda, candidates, jumpLengthCount * sizeof(cudaInstance), 
                cudaMemcpyHostToDevice);
    // printf("%u", __LINE__);
    // cudaInstanceVector* candidates;// = (cudaInstanceVector*) malloc(sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    // cudaMallocManaged(&candidates, sizeof(cudaInstanceVector) *static_cast<int>(neighbors->neighbors->size()));
    // cudaDeviceSynchronize();
    // int* sizeOfCandidates;// = (int*) malloc (sizeof(int) * neighbors->neighbors->size());
    // cudaMallocManaged(&sizeOfCandidates, sizeof(int) * static_cast<int>(neighbors->neighbors->size()));
    // cudaDeviceSynchronize();
    // for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
    //     cudaMallocManaged(&(candidates[i].instance), sizeof(cudaInstance) * static_cast<int>(neighbors->neighbors->operator[](i).size()));
    //     cudaDeviceSynchronize();
    //     // candidates[i].instance = (cudaInstance*) malloc(sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size());
    //     sizeOfCandidates[i] = static_cast<int>(neighbors->neighbors->operator[](i).size());
    //     for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
    //         candidates[i].instance[j].x = static_cast<int>(neighbors->neighbors->operator[](i)[j]);
    //         cudaDeviceSynchronize();
    //     }  
    // }
    // printf("%u", __LINE__);
    
    // printf("size %u", neighbors->neighbors->size());
    // cudaInstanceVector* h_data = (cudaInstanceVector*) malloc(sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    // cudaMemcpy(d_data, h_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyHostToDevice);
    // printf("h_data[i] list pointer adress: %u\n", candidates[0].instance);
    // printf("h_data[i] list pointer adress: %u\n", &(candidates[0].instance));
    
    // for (size_t i = 0; i < neighbors->neighbors->operator[](0).size(); ++i) {
    //     printf("h_data[i] list pointer adress: %u\n", candidates[0].instance[i].x);
    //     // printf("&h_data[i] list pointer adress: %u\n", &(h_data[i].instance));
    //     // printf("d_data[i] list pointer adress: %u\n", d_data[i]);
    //     // printf("&d_data[i] list pointer adress: %u\n", &(d_data[i]));
    // }
    // memcpy(h_data, candidates, sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    // for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        // printf("size: %u", neighbors->neighbors->operator[](i).size());
        // printf("sizeof(cudaInstance): %u", sizeof(cudaInstance));
        
        // cudaMalloc((void **) &(h_data[i].instance), sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size());
        // cudaMemcpy(h_data[i].instance, candidates[i].instance, sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size(), cudaMemcpyHostToDevice);
    // }
    
    // cudaInstanceVector* d_data;
    // cudaMalloc((void **) &d_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    // cudaMemcpy(d_data, h_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyHostToDevice);
    // for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
    //     printf("h_data[i] list pointer adress: %u\n", h_data[i].instance);
    //     printf("&h_data[i] list pointer adress: %u\n", &(h_data[i].instance));
    //     // printf("d_data[i] list pointer adress: %u\n", d_data[i]);
    //     // printf("&d_data[i] list pointer adress: %u\n", &(d_data[i]));
    // }
    // int* sizeOfCandidatesCuda;
    // int size = neighbors->neighbors->size();
    // // cudaMalloc((void **) &sizeOfCandidatesCuda, sizeof(int) * neighbors->neighbors->size());
    // // cudaMemcpy(sizeOfCandidatesCuda, sizeOfCandidates, sizeof(int) * neighbors->neighbors->size(), cudaMemcpyHostToDevice);
    
    // printf("Neihgbor Feature list pointer adress: %u\n", (*mDev_FeatureList));
    // printf("Neihgbor &Feature list pointer adress: %u\n", &(*mDev_FeatureList));
    // printf("Neihgbor pDevValueList pointer adress: %u\n", (*mDev_ValuesList));
    // printf("Neihgbor &pDevValueList pointer adress: %u\n", &(*mDev_ValuesList));
    // printf("Neihgbor Inverse size pointer adress: %u\n", (*mDev_SizeOfInstanceList));
    // printf("Neihgbor Inverse &size pointer adress: %u\n", &(*mDev_SizeOfInstanceList));
    // printf("Neihgbor jumppointer adress: %u\n", (*mDev_JumpLengthList));
    // printf("Neihgbor &jump pointer adress: %u\n", &(*mDev_JumpLengthList));
    if (pSimilarity) {
        // cosineSimilarityCuda<<<32, 96>>>(d_data, size, sizeOfCandidatesCuda, mDev_FeatureList,
        //                                 mDev_ValuesList, mDev_SizeOfInstanceList, mMaxNnz);
    } else {
    printf("%u\n", __LINE__);
    fflush(stdout);  
        euclideanDistanceCuda<<<128, 128>>>(candidatesCuda, jumpLengthCuda, numberOfCandidatesPerInstanceCuda, 
                                        neighbors->neighbors->size(), (*mDev_FeatureList),
                                        (*mDev_ValuesList), (*mDev_SizeOfInstanceList), 
                                        (*mDev_JumpLengthList), (*mDev_DotProducts));
        cudaDeviceSynchronize();
    printf("%u\n", __LINE__);
    fflush(stdout); 
        
    }
    printf("%u\n", __LINE__);
    fflush(stdout); 
    
    printf("75");
    fflush(stdout);
    
    // cudaMemcpy(h_data, d_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyDeviceToHost);
    // printf("78");
    // fflush(stdout);
    
    // for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
    //     cudaMemcpy((void **) candidates[i].instance, h_data[i].instance, sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size(), cudaMemcpyDeviceToHost);
        // cudaFree(candidates[i].instance);
    // }
    // cudaFree(candidates)
    // printf("84");
    // fflush(stdout);
    
    // memcpy(candidates, h_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    // cudaMemcpy(candidates, d_data, sizeof(cudaInstanceVector) * neighbors->neighbors->size(), cudaMemcpyDeviceToHost);
    // printf("88");
    // fflush(stdout);
    
    // cudaFree(d_data);
    // cudaInstance* candidatesInstance = (cudaInstance* )
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        // cudaMallocManaged(&(candidates[i].instance), static_cast<int>(neighbors->neighbors->operator[](i).size()));
        // cudaDeviceSynchronize();
        // candidates2[i].instance = (cudaInstance*) malloc(sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size());
        // sizeOfCandidates[i] = static_cast<int>(neighbors->neighbors->operator[](i).size());
        cudaMemcpy(candidates, candidatesCuda, sizeof(cudaInstance) * jumpLengthCount, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // candidates2[i].instance
        // for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
        //     candidates2[i].instance[j].x = candidates[i].instance[j].x;
        //     candidates2[i].instance[j].y = candidates[i].instance[j].y;
            
        //     cudaDeviceSynchronize();
        // }  
    }
    printf("%u\n", __LINE__);
    fflush(stdout);
    cudaInstanceVector* candidates2 = (cudaInstanceVector*) malloc(sizeof(cudaInstanceVector) * neighbors->neighbors->size());
    // cudaInstance* candidatesLocal;
    printf("%u\n", __LINE__);
    fflush(stdout);
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        // cudaMallocManaged(&(candidates[i].instance), sizeof(cudaInstance) * static_cast<int>(neighbors->neighbors->operator[](i).size()));
        // cudaDeviceSynchronize();
         candidates2[i].instance = (cudaInstance*) malloc(sizeof(cudaInstance) * neighbors->neighbors->operator[](i).size());
        // sizeOfCandidates[i] = static_cast<int>(neighbors->neighbors->operator[](i).size());
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            candidates2[i].instance[j].x = candidates[jumpLength[i]+j].x;
            candidates2[i].instance[j].y = candidates[jumpLength[i]+j].y;
            
            // cudaDeviceSynchronize();
        }  
    }
    // cudaFree(sizeOfCandidates);
    
    // free(h_data);
    // free(sizeOfCandidates);
    printf("94");
    fflush(stdout);
    
    return candidates2;
}