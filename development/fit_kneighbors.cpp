#include <vector>
#include <iostream>
#include <iterator>
#include <fstream>
#include <algorithm>
#include <string>

#include <math.h>

#include "minHash.h"

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
    	dataValue = (*pFeatures)[i];

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

MinHash* createMinHashObj(size_t pNumberOfHashFunctions, size_t pBlockSize, size_t pNumberOfCores, 
							size_t pChunkSize, size_t pNneighbors, size_t pMinimalBlocksInCommon, 
							size_t pMaxBinSize,  size_t pMaximalNumberOfHashCollisions, 
							size_t pExcessFactor, int pFast) {
	MinHash* minHashObj = new MinHash (pNumberOfHashFunctions, pBlockSize, pNumberOfCores, pChunkSize,
                    pMaxBinSize, pNneighbors, pMinimalBlocksInCommon, 
                    pExcessFactor, pMaximalNumberOfHashCollisions, pFast);
	return minHashObj;
}

void fit(MinHash* minHashObj, std::vector<size_t>* pInstances, std::vector<size_t>* pFeatures, std::vector<float>* pData,
                                              size_t pMaxNumberOfInstances, size_t pMaxNumberOfFeatures) {
	SparseMatrixFloat* originalDataMatrix = parseRawData(pInstances, pFeatures, pData, 
                                                    pMaxNumberOfInstances, pMaxNumberOfFeatures);
	minHashObj->set_mOriginalData(originalDataMatrix);
	minHashObj->fit(originalDataMatrix);
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

	std::istream_iterator<float> begin_addInfo(file_addInfo);
    std::istream_iterator<float> end_addInfo; 
    std::copy(begin_addInfo, end_addInfo, std::back_inserter(addInfo));

    size_t numberOfHashFunctions = 400;
    size_t blockSize = 4;
    size_t numberOfCores = 4; 
    size_t chunkSize = 0;
    size_t nNeighbors = 5;
    size_t minimalBlocksInCommon = 1; 
    size_t maxBinSize = 50;
    size_t maximalNumberOfHashCollisions = ceil(numberOfHashFunctions / static_cast<float>(blockSize));
    size_t excessFactor = 5;
    int fast = 0;

    MinHash* minHash = createMinHashObj(numberOfHashFunctions, blockSize, numberOfCores, chunkSize, nNeighbors,
    					minimalBlocksInCommon, maxBinSize, maximalNumberOfHashCollisions,
    					excessFactor, fast);
    fit(minHash, &instances, &features, &data, addInfo[0], addInfo[1]);
    SparseMatrixFloat* dataSparse = new SparseMatrixFloat(0, 0);
    neighborhood* neighborhood_ = minHash->kneighbors(dataSparse, nNeighbors, fast);

    std::cout << "\n[";
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
	return 0;
}