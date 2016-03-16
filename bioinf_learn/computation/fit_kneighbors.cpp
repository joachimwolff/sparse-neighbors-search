#include <vector>
#include <iostream>
#include <iterator>
#include <fstream>
#include <algorithm>
#include <string>

#include <math.h>

#include "nearestNeighbors.h"

SparseMatrixFloat* parseRawData(std::vector<size_t>* pInstances, std::vector<size_t>* pFeatures, std::vector<float>* pData,
                                              size_t pMaxNumberOfInstances, size_t pMaxNumberOfFeatures) {

    size_t instanceOld = 0;
    SparseMatrixFloat* originalData = new SparseMatrixFloat(pMaxNumberOfInstances, pMaxNumberOfFeatures);
    size_t featuresCount = 0;
    size_t featureValue;
    size_t instanceValue;
    float dataValue;
    for (size_t i = 0; i < pInstances->size(); ++i) {
        
    	instanceValue = (*pInstances)[i];
    	featureValue = (*pFeatures)[i];
    	dataValue = (*pData)[i];

        if (instanceOld != instanceValue) {
            originalData->insertToSizesOfInstances(instanceOld, featuresCount);
            featuresCount = 0;
        }
        originalData->insertElement(instanceValue, featuresCount, featureValue, dataValue);
        instanceOld = instanceValue;
        ++featuresCount;
    }
    originalData->insertToSizesOfInstances(instanceOld, featuresCount);

    return originalData;
}

NearestNeighbors* createNearestNeighborsObj(size_t pNumberOfHashFunctions, size_t pShingleSize, size_t pNumberOfCores, 
							size_t pChunkSize, size_t pNneighbors, size_t pMinimalBlocksInCommon, 
							size_t pMaxBinSize,  size_t pMaximalNumberOfHashCollisions, 
							size_t pExcessFactor, int pFast, int pSimilarity, size_t pBloomierFilter,
                            size_t pPrune_inverse_index, float pPrune_inverse_index_after_instance, 
                            int pRemoveHashFunctionWithLessEntriesAs,
                            size_t pHash_algorithm, size_t pBlock_size, size_t pShingle, 
                            size_t pRemoveValueWithLeastSigificantBit) {
	NearestNeighbors* nearestNeighborsObj = new NearestNeighbors (pNumberOfHashFunctions, pShingleSize, pNumberOfCores, pChunkSize,
                    pMaxBinSize, pNneighbors, pMinimalBlocksInCommon, 
                    pExcessFactor, pMaximalNumberOfHashCollisions, pFast, pSimilarity, pBloomierFilter,
                    pPrune_inverse_index, pPrune_inverse_index_after_instance,
                    pRemoveHashFunctionWithLessEntriesAs, pHash_algorithm,
                    pBlock_size, pShingle, pRemoveValueWithLeastSigificantBit);
	return nearestNeighborsObj;
}

void fit(NearestNeighbors* nearestNeighborsObj, std::vector<size_t>* pInstances, std::vector<size_t>* pFeatures, std::vector<float>* pData,
                                              size_t pMaxNumberOfInstances, size_t pMaxNumberOfFeatures) {
    // std::cout << __LINE__ << std::endl;
                                                  
	SparseMatrixFloat* originalDataMatrix = parseRawData(pInstances, pFeatures, pData, 
                                                    pMaxNumberOfInstances, pMaxNumberOfFeatures);
    // std::cout << __LINE__ << std::endl;
                                                    
	nearestNeighborsObj->set_mOriginalData(originalDataMatrix);
    // std::cout << __LINE__ << std::endl;
    
	nearestNeighborsObj->fit(originalDataMatrix);
    // std::cout << __LINE__ << std::endl;
    
}
 
   

    
int main( int argc, const char* argv[] ) { 
	std::string fileNameStr("bursi");
    std::string instancesString("_instances");
    std::string featuresString("_features");
    std::string dataString("_data");
    std::string additionalInformation("_addInfo");
	std::vector<size_t> instances;
    std::vector<size_t> features;
    std::vector<float> data;
    std::vector<size_t> addInfo;

    std::fstream file_instances(fileNameStr+instancesString, std::ios::in);
    std::fstream file_features(fileNameStr+featuresString, std::ios::in);
    std::fstream file_data(fileNameStr+dataString, std::ios::in);
    std::fstream file_addInfo(fileNameStr+additionalInformation, std::ios::in);

	std::istream_iterator<size_t> begin_instances(file_instances);
    std::istream_iterator<size_t> end_instances; 
    std::copy(begin_instances,end_instances,std::back_inserter(instances));

    std::istream_iterator<size_t> begin_features(file_features);
    std::istream_iterator<size_t> end_features; 
    std::copy(begin_features, end_features, std::back_inserter(features));

    std::istream_iterator<float> begin_data(file_data);
    std::istream_iterator<float> end_data; 
    std::copy(begin_data, end_data, std::back_inserter(data));

	std::istream_iterator<size_t> begin_addInfo(file_addInfo);
    std::istream_iterator<size_t> end_addInfo; 
    std::copy(begin_addInfo, end_addInfo, std::back_inserter(addInfo));

    size_t numberOfHashFunctions = 150;
    size_t shingleSize = 4;
    
    size_t numberOfCores = 4; 
    size_t chunkSize = 0;
    size_t nNeighbors = 5;
    size_t minimalBlocksInCommon = 1; 
    size_t maxBinSize = 50;
    size_t maximalNumberOfHashCollisions = ceil(numberOfHashFunctions / static_cast<float>(shingleSize));
    size_t excessFactor = 11;
    int fast = 0;
    int similarity = 0;
    size_t bloomierFilter = 0;
    int prune_inverse_index=2;
    float prune_inverse_index_after_instance=0.5;
    float removeHashFunctionWithLessEntriesAs=0;
    size_t hash_algorithm = 0;
    size_t block_size = 1;
    size_t shingle=0;



    // std::cout << __LINE__ << std::endl;
    NearestNeighbors* nearestNeighbors = createNearestNeighborsObj(numberOfHashFunctions, shingleSize, numberOfCores, chunkSize, nNeighbors,
    					minimalBlocksInCommon, maxBinSize, maximalNumberOfHashCollisions,
    					excessFactor, fast, similarity, bloomierFilter, prune_inverse_index, 
                                                    prune_inverse_index_after_instance, removeHashFunctionWithLessEntriesAs,
                                                    hash_algorithm, block_size, shingle, 0);
    // std::cout << __LINE__ << std::endl;
                                                    
    fit(nearestNeighbors, &instances, &features, &data, addInfo[0], addInfo[1]);
    // std::cout << __LINE__ << std::endl;
    
    SparseMatrixFloat* dataSparse = parseRawData(&instances, &features, &data, 
                                                  addInfo[0], addInfo[1]);
    // std::cout << __LINE__ << std::endl;
                                                  
    neighborhood* neighborhood_ = nearestNeighbors->kneighbors(dataSparse, nNeighbors, fast);
    // neighborhood* neighborhood2_ = nearestNeighbors->kneighbors(dataSparse, nNeighbors, fast);
    // std::cout << "\n[";
    for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {
    	std::cout << "[";
    	for (size_t j = 0; j < neighborhood_->neighbors->operator[](i).size(); ++j) {
    		std::cout << " " << neighborhood_->neighbors->operator[](i)[j];
    	}
    	std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;

	std::cout << "\n[";
    for (size_t i = 0; i < neighborhood_->distances->size(); ++i) {
    	std::cout << "[";
    	for (size_t j = 0; j < neighborhood_->distances->operator[](i).size(); ++j) {
    		std::cout << " " << neighborhood_->distances->operator[](i)[j];
    	}
    	std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
    delete nearestNeighbors;
    delete dataSparse;
    delete neighborhood_->neighbors;
    delete neighborhood_->distances;
    delete neighborhood_;
    
	return 0;
}