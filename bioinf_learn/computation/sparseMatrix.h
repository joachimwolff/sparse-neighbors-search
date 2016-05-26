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
#include <xmmintrin.h>
// #include <stdio.h>
#include <cstring>
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
    int* mSparseMatrix;
    float*  mSparseMatrixValues;
    size_t* mSizesOfInstances;
    
    size_t mMaxNnz;
    size_t mNumberOfInstances;
   
    std::unordered_map<size_t, float> mDotProductPrecomputed;
  public:
    SparseMatrixFloat(size_t pNumberOfInstances, size_t pMaxNnz) {
        
        pMaxNnz = pMaxNnz + 32 - (pMaxNnz % 32);
        mSparseMatrix = new int [pNumberOfInstances * pMaxNnz];
        std::fill_n(mSparseMatrix, pNumberOfInstances * pMaxNnz, MAX_VALUE);
        mSparseMatrixValues = new float [pNumberOfInstances * pMaxNnz]();
        mSizesOfInstances = new size_t [pNumberOfInstances];
        mMaxNnz = pMaxNnz;
        mNumberOfInstances = pNumberOfInstances;
    };
    ~SparseMatrixFloat() {
        delete [] mSparseMatrix;
        delete [] mSparseMatrixValues;
        delete [] mSizesOfInstances;
    };
    
    void precomputeDotProduct() {
        
        double value = 0.0;
        double value1 = 0.0;
        double value2 = 0.0;
        double value3 = 0.0;
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
            
            mDotProductPrecomputed[i] = (float) value;
            value = 0.0;
            value1 = 0.0;
            value2 = 0.0;
            value3 = 0.0;
        }
    };
    float dotProduct(const size_t pIndex, const size_t pIndexNeighbor, SparseMatrixFloat* pQueryData=NULL)  {
        SparseMatrixFloat* queryData = this;
         if (pQueryData != NULL) {
            queryData = pQueryData;
        }
        double value = 0.0;  
        size_t counterInstance = 0;
        size_t counterNeighbor = 0;
        size_t sizeInstance = queryData->getSizeOfInstance(pIndex);
        size_t sizeNeighbor = this->getSizeOfInstance(pIndexNeighbor);
       
        while (counterInstance <  sizeInstance && counterNeighbor < sizeNeighbor) {
           
            if (queryData->getNextElement(pIndex, counterInstance) < this->getNextElement(pIndexNeighbor, counterNeighbor)) {
                ++counterInstance;
            } else if (queryData->getNextElement(pIndex, counterInstance) > this->getNextElement(pIndexNeighbor, counterNeighbor)){
                ++counterNeighbor;
            } else {
                value += (double) queryData->getNextValue(pIndex, counterInstance) * (double) this->getNextValue(pIndexNeighbor, counterNeighbor);
                ++counterInstance;
                ++counterNeighbor;
            }
        }
        return (float) value; 
    };
    float getDotProductPrecomputed(size_t pIndex) {
        return mDotProductPrecomputed[pIndex];
    }
    int* getSparseMatrixIndex() const{
        return mSparseMatrix;
    };
    float* getSparseMatrixValues() const{
        return mSparseMatrixValues;
    };
    
    
    int* getSparseMatrixIndexPointer(size_t pIndex) {
        // printf("pIndex: %u, mMaxNNZ: %u pIndex*mMaxNnz: %u maxIndex: %u\n", pIndex, mMaxNnz, pIndex * mMaxNnz, mNumberOfInstances * mMaxNnz);
        
        return &(mSparseMatrix[pIndex * mMaxNnz]);
    };
    float* getSparseMatrixValuesPointer(size_t pIndex) {
        // printf("pIndex: %u, mMaxNNZ: %u pIndex*mMaxNnz: %u maxIndex: %u\n", pIndex, mMaxNnz, pIndex * mMaxNnz, mNumberOfInstances * mMaxNnz);
        return &(mSparseMatrixValues[pIndex * mMaxNnz]);
    };
    
    size_t* getSparseMatrixSizeOfInstances() const {
        return mSizesOfInstances;
    };
    
    size_t getMaxNnz() const {
        return mMaxNnz;
    };
    size_t getNumberOfInstances() const {
        return mNumberOfInstances;
    };
    int getNextElement(size_t pInstance, size_t pCounter) const {
                return mSparseMatrix[pInstance*mMaxNnz+pCounter];
    };
    float getNextValue(size_t pInstance, size_t pCounter) const {
                return mSparseMatrixValues[pInstance*mMaxNnz+pCounter];
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
            mSparseMatrix[pInstanceId*mMaxNnz + pNnzCount] = static_cast<int> (pFeatureId);
            mSparseMatrixValues[pInstanceId*mMaxNnz + pNnzCount] = pValue;
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
        
        int* tmp_mSparseMatrix = new int [numberOfInstances * maxNnz];
        float* tmp_mSparseMatrixValues = new float [numberOfInstances * maxNnz];
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
        
        const size_t valueXX = getDotProductPrecomputed(pRowId);
        
        float valueXY = 0;
        float valueYY = 0;
        size_t instance_id;
        for (size_t i = 0; i < pRowIdVector.size(); ++i) {
            instance_id = pRowIdVector[i];
            sortMapFloat element; 
            element.key = pRowIdVector[i];
            element.val = 0;
            
            valueXY = this->dotProduct(pRowId, instance_id, pQueryData);
            if (pQueryData == NULL) {
                // std::cout << "instance_id: " << instance_id << std::endl;
                valueYY = getDotProductPrecomputed(instance_id);
                // std::cout << "\tDONE" << std::endl;
            } else {
                // std::cout << "instance_id: " << instance_id << std::endl;
                
                valueYY = pQueryData->getDotProductPrecomputed(instance_id);
                // std::cout << "\tDONE" << std::endl;
                
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
            // std::cout << __LINE__ << std::endl;

        std::partial_sort(returnValue.begin(), returnValue.begin()+numberOfElementsToSort, returnValue.end(), mapSortAscByValueFloat);
        //   std::cout << __LINE__ << std::endl;

        return returnValue;
    };

    std::vector<sortMapFloat> cosineSimilarity(const std::vector<size_t> pRowIdVector, const size_t pNneighbors, 
                                                const size_t pQueryId, SparseMatrixFloat* pQueryData=NULL)  {
       
        const size_t pRowId = pQueryId;
        
        std::vector<sortMapFloat> returnValue(pRowIdVector.size());
        
        const size_t valueXX = getDotProductPrecomputed(pRowId);
        
        float valueXY = 0;
        float valueYY = 0;
        size_t instance_id;
        for (size_t i = 0; i < pRowIdVector.size(); ++i) {
            instance_id = pRowIdVector[i];
            sortMapFloat element; 
            element.key = pRowIdVector[i];
            element.val = 0;
            
            valueXY = this->dotProduct(pRowId, instance_id, pQueryData);
            if (pQueryData == NULL) {
                valueYY = getDotProductPrecomputed(instance_id);
            } else {
                valueYY = pQueryData->getDotProductPrecomputed(instance_id);
            }
            element.val = valueXY / (sqrt(valueXX) * sqrtf(valueYY));
            
            if (element.val <= 0) {
                element.val = 0;
            }
                
            returnValue[i] = element;
        }
        size_t numberOfElementsToSort = pNneighbors;
        if (numberOfElementsToSort > returnValue.size()) {
            numberOfElementsToSort = returnValue.size();
        }
        // sort the values by decreasing order
        std::partial_sort(returnValue.begin(), returnValue.begin()+numberOfElementsToSort, returnValue.end(), mapSortDescByValueFloat);
        return returnValue;
    };
};
#endif // SPARSE_MATRIX_H