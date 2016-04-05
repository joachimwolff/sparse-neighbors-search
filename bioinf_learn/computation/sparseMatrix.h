/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
 Winter semester 2015/2016
 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/
#include <math.h>

#include <algorithm>
#include <iostream>
#include "typeDefinitionsBasic.h"

#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H


class SparseMatrixFloat {

  private: 
    // stores only the pointer addresses to:
    //      if even index: size_t which is pointer for vsize
    //      if odd indeX: size_t which is pointer for vfloat
    size_t* mSparseMatrix;
    size_t*  mSparseMatrixValues;
    size_t* mSizesOfInstances;
    size_t mMaxNnz;
    size_t mNumberOfInstances;
    std::unordered_map<size_t, size_t> mDotProductPrecomputed;
  public:
    SparseMatrixFloat(size_t pNumberOfInstances, size_t pMaxNnz) {
        mSparseMatrix = new size_t [pNumberOfInstances * pMaxNnz];
        mSparseMatrixValues = new size_t [pNumberOfInstances * pMaxNnz];
        mSizesOfInstances = new size_t [pNumberOfInstances];
        // for (size_t i = 0; i < pNumberOfInstances; ++i) {
        //     mSizesOfInstances[i] = 0;
        // }
        mMaxNnz = pMaxNnz;
        mNumberOfInstances = pNumberOfInstances;
    };
    ~SparseMatrixFloat() {
        delete [] mSparseMatrix;
        delete [] mSparseMatrixValues;
        delete [] mSizesOfInstances;
    };
    
    void precomputeDotProduct() {
        // std::cout << __LINE__ << std::endl;
        
        size_t value = 0;
        size_t value1 = 0;
        size_t value2 = 0;
        size_t value3 = 0;
        for (size_t i = 0; i < size(); ++i) {
            size_t j = 0;
            for (j = 0; j < getSizeOfInstance(i); j += 4) {
                value += pow(getNextValue(i, j), 2);
                value1 += pow(getNextValue(i, j+1), 2);
                value2 += pow(getNextValue(i, j+2), 2);
                value3 += pow(getNextValue(i, j+3), 2);
            }
            while (j < getSizeOfInstance(i)) {
                value += pow(getNextValue(i, j), 2);
                ++j;
            }
            value = value + value1 + value2 + value3;
            mDotProductPrecomputed[i] = value;
            value = 0;
            value1 = 0;
            value2 = 0;
            value3 = 0;
            // std::cout << "instance: " << i << ": value: " << mDotProductPrecomputed[i] << std::endl;
        }
        // std::cout << __LINE__ << std::endl;
        
    };
    float dotProduct(const size_t pIndex, const size_t pIndexNeighbor, SparseMatrixFloat* pQueryData=NULL)  {
        SparseMatrixFloat* queryData = this;
         if (pQueryData != NULL) {
            queryData = pQueryData;
        }
        size_t value = 0;  
        size_t counterInstance = 0;
        size_t counterNeighbor = 0;
        size_t sizeInstance = queryData->getSizeOfInstance(pIndex);
        size_t sizeNeighbor = this->getSizeOfInstance(pIndexNeighbor);
        // std::vector<sparseData>* instance = queryData->getDataMatrix()->operator[](pIndex);
        // std::vector<sparseData>* neighbor = this->getDataMatrix()->operator[](pIndexNeighbor);
       
        // auto iteratorInstance = instance->begin();
        // auto iteratorNeighbor = neighbor->begin();
        while (counterInstance <  sizeInstance && counterNeighbor < sizeNeighbor) {
            if (queryData->getNextElement(pIndex, counterInstance) < this->getNextElement(pIndexNeighbor, counterNeighbor)) {
                ++counterInstance;
            } else if (queryData->getNextElement(pIndex, counterInstance) > this->getNextElement(pIndexNeighbor, counterNeighbor)){
                ++counterNeighbor;
            } else {
                value += queryData->getNextValue(pIndex, counterInstance) * this->getNextValue(pIndexNeighbor, counterNeighbor);
                ++counterInstance;
                ++counterNeighbor;
            }
        }
        return value; 
    };
    size_t getDotProductPrecomputed(size_t pIndex) {
        return mDotProductPrecomputed[pIndex];
    }
    size_t* getSparseMatrixIndex() const{
        return mSparseMatrix;
    };
    size_t* getSparseMatrixValues() const{
        return mSparseMatrixValues;
    };
    size_t* getSparseMatrixSizeOfInstances() const{
        return mSizesOfInstances;
    };
    size_t getMaxNnz() const {
        return mMaxNnz;
    };
    size_t getNumberOfInstances() const {
        return mNumberOfInstances;
    };
    size_t getNextElement(size_t pInstance, size_t pCounter) const {
        if (pInstance < mNumberOfInstances && pInstance*mMaxNnz+pCounter < mNumberOfInstances*mMaxNnz) {
            if (pCounter < mSizesOfInstances[pInstance]) {
                return mSparseMatrix[pInstance*mMaxNnz+pCounter];
            }
        }
        return MAX_VALUE;
    };
    size_t getNextValue(size_t pInstance, size_t pCounter) const {
        if (pInstance < mNumberOfInstances && pInstance*mMaxNnz+pCounter < mNumberOfInstances*mMaxNnz) {
            if (pCounter < mSizesOfInstances[pInstance]) {
                return mSparseMatrixValues[pInstance*mMaxNnz+pCounter];
            }
        }
        return MAX_VALUE;
    };
    size_t size() const {
        return mNumberOfInstances;
    };
    
    size_t getSizeOfInstance(size_t pInstance) const {
        if (pInstance < mNumberOfInstances) {
            return mSizesOfInstances[pInstance];
        }
        return 0;
    };
    void insertElement(size_t pInstanceId, size_t pNnzCount, size_t pFeatureId, float pValue) {
        if (pInstanceId*mMaxNnz + pNnzCount < mNumberOfInstances * mMaxNnz) {
            mSparseMatrix[pInstanceId*mMaxNnz + pNnzCount] = pFeatureId;
            mSparseMatrixValues[pInstanceId*mMaxNnz + pNnzCount] = (size_t) pValue*1000;
        }
    };

    void insertToSizesOfInstances(size_t pInstanceId, size_t pSizeOfInstance) {
        if (pInstanceId < mNumberOfInstances) {
            mSizesOfInstances[pInstanceId] = pSizeOfInstance;
        }
    };
    void addNewInstancesPartialFit(const SparseMatrixFloat* pMatrix) {
        size_t numberOfInstances = this->getNumberOfInstances();
        numberOfInstances += pMatrix->getNumberOfInstances();
        size_t maxNnz = std::max(mMaxNnz, pMatrix->getMaxNnz());
        
        size_t* tmp_mSparseMatrix = new size_t [numberOfInstances * maxNnz];
        size_t* tmp_mSparseMatrixValues = new size_t [numberOfInstances * maxNnz];
        size_t* tmp_mSizesOfInstances = new size_t [numberOfInstances];
        
        for (size_t i = 0; i < this->getNumberOfInstances(); ++i) {
            for (size_t j = 0; j < this->getSizeOfInstance(i); ++j) {
                tmp_mSparseMatrix[i*maxNnz + j] = this->getNextElement(i, j);
                tmp_mSparseMatrixValues[i*maxNnz + j] = this->getNextValue(i,j);
            }
            tmp_mSizesOfInstances[i] = this->getSizeOfInstance(i);
        }
        for (size_t i = 0; i < pMatrix->getNumberOfInstances(); ++i) {
            for (size_t j = 0; j < pMatrix->getSizeOfInstance(i); ++j) {
                tmp_mSparseMatrix[(i+this->getNumberOfInstances())*maxNnz + j] = pMatrix->getNextElement(i, j);
                tmp_mSparseMatrixValues[(i+this->getNumberOfInstances())*maxNnz + j] = pMatrix->getNextValue(i,j);
            }
            tmp_mSizesOfInstances[i+this->getNumberOfInstances()] = pMatrix->getSizeOfInstance(i);
        }
        mMaxNnz = maxNnz;
        mNumberOfInstances = numberOfInstances;
        delete [] mSparseMatrix;
        delete [] mSparseMatrixValues;
        delete [] mSizesOfInstances;
        delete pMatrix;
        mSparseMatrix = tmp_mSparseMatrix;
        mSparseMatrixValues = tmp_mSparseMatrixValues;
        mSizesOfInstances = tmp_mSizesOfInstances;        
    };
    std::vector<sortMapFloat> euclidianDistance(const std::vector<size_t> pRowIdVector, const size_t pNneighbors, 
                                                const size_t pQueryId, SparseMatrixFloat* pQueryData=NULL) {
        
        const size_t pRowId = pQueryId;
        
        std::vector<sortMapFloat> returnValue(pRowIdVector.size());
        // std::cout << __LINE__ << std::endl;
        
        const size_t valueXX = getDotProductPrecomputed(pRowId);
        // std::cout << __LINE__ << std::endl;
        
        size_t valueXY = 0;
        size_t valueYY = 0;
        size_t instance_id;
        // uint32_t neighbor_id = 
        for (size_t i = 0; i < pRowIdVector.size(); ++i) {
            instance_id = pRowIdVector[i];
            sortMapFloat element; 
            element.key = pRowIdVector[i];
            element.val = 0.0;
            
            // if (pRowId < pRowIdVector[i]) {
            //     indexOuter = pRowId;
            //     indexInner = pRowIdVector[i];
            // } else {
            //     indexOuter = pRowIdVector[i];
            //     indexInner = pRowId;
            // }
        // std::cout << __LINE__ << std::endl;
            
            // auto iteratorEuclidDistance = (*mValuesPrecomputed)[indexOuter]->find(indexInner);
        // std::cout << __LINE__ << std::endl;
            
            // if (iteratorEuclidDistance != (*mValuesPrecomputed)[indexOuter]->end()) {
            //     element.val = (*iteratorEuclidDistance).second;
            // } else {
        // std::cout << __LINE__ << std::endl;
                
                valueXY = this->dotProduct(pRowId, instance_id, pQueryData);
                if (pQueryData == NULL) {
                    valueYY = getDotProductPrecomputed(instance_id);
                } else {
                    valueYY = pQueryData->getDotProductPrecomputed(instance_id);
                }
                element.val = valueXX - 2* valueXY + valueYY;
                if (element.val <= 0) {
                    element.val = 0;
                }
                
            returnValue[i] = element;
        }
        size_t numberOfElementsToSort = pNneighbors;
        if (numberOfElementsToSort > returnValue.size()) {
            numberOfElementsToSort = returnValue.size();
        }
        // sort the values by increasing order
        std::partial_sort(returnValue.begin(), returnValue.begin()+numberOfElementsToSort, returnValue.end(), mapSortAscByValueFloat);
        return returnValue;
    };

    std::vector<sortMapFloat> cosineSimilarity(const std::vector<size_t> pRowIdVector, const size_t pNneighbors, 
                                                const size_t pQueryId, const SparseMatrixFloat* pQueryData=NULL) const {
        std::cout << "cosine, queryId: " << pQueryId << std::endl;
       
        const SparseMatrixFloat* queryData = this;
        const size_t pRowId = pQueryId;
        
        if (pQueryData != NULL) {
            queryData = pQueryData;
        }
        std::vector<sortMapFloat> returnValue(pRowIdVector.size());
        size_t pointerToMatrixElement = 0;
        size_t pointerToVectorElement = 0;
        // iterate over all candidates

        for (size_t i = 0; i < pRowIdVector.size(); ++i) {
            sortMapFloat element; 
            element.key = pRowIdVector[i];
            element.val = 0;
            size_t dotProduct = 0;
            size_t magnitudeFirstVector = 0;
            size_t magnitudeSecondVector = 0;

            // features ids are stored in mSparseMatrix. 
            // every instances starts at index indexId*mMaxNnz --> every instance can store maximal mMaxNnz feature ids
            // how many elements per index are stored is stored in mSizesOfInstances[indexID]
            // iterate until both instances have no more feature ids
            bool endOfFirstVectorNotReached = pointerToMatrixElement < this->getSizeOfInstance(pRowIdVector[i]);
            bool endOfSecondVectorNotReached = pointerToVectorElement < queryData->getSizeOfInstance(pRowId);

            while (endOfFirstVectorNotReached && endOfSecondVectorNotReached) {
                // are the feature ids of the two instances the same?
                size_t featureIdFirstVector = this->getNextElement(pRowIdVector[i], pointerToMatrixElement);
                size_t featureIdSecondVector = queryData->getNextElement(pRowId, pointerToVectorElement);

                if (featureIdFirstVector == featureIdSecondVector) {
                    // if they are the same substract the values, compute the square and sum it up
                    dotProduct += this->getNextValue(pRowIdVector[i], pointerToMatrixElement) * queryData->getNextValue(pRowId, pointerToVectorElement);
                    magnitudeFirstVector += pow(this->getNextValue(pRowIdVector[i], pointerToMatrixElement), 2);
                    magnitudeSecondVector += pow(queryData->getNextValue(pRowId, pointerToVectorElement),2);
                    // increase both counters to the next element 
                    ++pointerToMatrixElement;
                    ++pointerToVectorElement;
                } else if (featureIdFirstVector < featureIdSecondVector) {
                    // if the feature ids are unequal square only the smaller one and add it to the sum
                    magnitudeFirstVector += pow(this->getNextValue(pRowIdVector[i], pointerToMatrixElement), 2);
                    // increase counter for first vector
                    ++pointerToMatrixElement;
                } else {
                    magnitudeSecondVector += pow(queryData->getNextValue(pRowId, pointerToVectorElement), 2);
                    // increase counter for second vector
                    ++pointerToVectorElement;
                }
                endOfFirstVectorNotReached = pointerToMatrixElement < this->getSizeOfInstance(pRowIdVector[i]);
                endOfSecondVectorNotReached = pointerToVectorElement < queryData->getSizeOfInstance(pRowId);
            }
            while (endOfFirstVectorNotReached) {
                magnitudeFirstVector += pow(this->getNextValue(pRowIdVector[i], pointerToMatrixElement), 2);
                ++pointerToMatrixElement;
                endOfFirstVectorNotReached = pointerToMatrixElement < this->getSizeOfInstance(pRowIdVector[i]);
            }
            while (endOfSecondVectorNotReached) {
                magnitudeSecondVector += pow(queryData->getNextValue(pRowId, pointerToVectorElement), 2);
                ++pointerToVectorElement;
                endOfSecondVectorNotReached = pointerToVectorElement < queryData->getSizeOfInstance(pRowId);
            }

            pointerToMatrixElement = 0;
            pointerToVectorElement = 0;
            // compute cosine similarity
            element.val = dotProduct / (magnitudeFirstVector * magnitudeSecondVector);
            
            returnValue[i] = element;
        }
        size_t numberOfElementsToSort = pNneighbors;
        if (numberOfElementsToSort > returnValue.size()) {
            numberOfElementsToSort = returnValue.size();
        }
        // sort the values by increasing order
        // std::sort(returnValue.begin(), returnValue.end(), mapSortDescByValueFloat);
        
        std::partial_sort(returnValue.begin(), returnValue.begin()+numberOfElementsToSort, returnValue.end(), mapSortDescByValueFloat);
        return returnValue;
    };
};
#endif // SPARSE_MATRIX_H