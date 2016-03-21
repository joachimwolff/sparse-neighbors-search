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
    float*  mSparseMatrixValues;
    size_t* mSizesOfInstances;
    size_t mMaxNnz;
    size_t mNumberOfInstances;
  public:
    SparseMatrixFloat(size_t pNumberOfInstances, size_t pMaxNnz) {
        mSparseMatrix = new size_t [pNumberOfInstances * pMaxNnz];
        mSparseMatrixValues = new float [pNumberOfInstances * pMaxNnz];
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
    size_t* getSparseMatrixIndex() const{
        return mSparseMatrix;
    };
    float* getSparseMatrixValues() const{
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
    float getNextValue(size_t pInstance, size_t pCounter) const {
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
        
        size_t* tmp_mSparseMatrix = new size_t [numberOfInstances * maxNnz];
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
    std::vector<sortMapFloat> euclidianDistance(const std::vector<int> pRowIdVector, const size_t pNneighbors, 
                                                const size_t pQueryId, const SparseMatrixFloat* pQueryData=NULL) const {
        // std::cout << "euclidean distance, queryId: " << pQueryId << std::endl;
        
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
            element.val = 0.0;
            // std::cout << pRowIdVector[i];
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
                    float value = this->getNextValue(pRowIdVector[i], pointerToMatrixElement) - queryData->getNextValue(pRowId, pointerToVectorElement);
                    element.val += value * value;
                    // increase both counters to the next element 
                    ++pointerToMatrixElement;
                    ++pointerToVectorElement;
                } else if (featureIdFirstVector < featureIdSecondVector) {
                    // if the feature ids are unequal square only the smaller one and add it to the sum
                    float value = this->getNextValue(pRowIdVector[i], pointerToMatrixElement);
                    element.val += value * value;
                    // increase counter for first vector
                    ++pointerToMatrixElement;
                } else {
                    float value = queryData->getNextValue(pRowId, pointerToVectorElement);
                    element.val += value * value;
                    ++pointerToVectorElement;
                }

                endOfFirstVectorNotReached = pointerToMatrixElement < this->getSizeOfInstance(pRowIdVector[i]);
                endOfSecondVectorNotReached = pointerToVectorElement < queryData->getSizeOfInstance(pRowId);
            }
            while (endOfFirstVectorNotReached) {
                float value = this->getNextValue(pRowIdVector[i], pointerToMatrixElement);
                element.val += value * value;
                ++pointerToMatrixElement;
                endOfFirstVectorNotReached = pointerToMatrixElement < this->getSizeOfInstance(pRowIdVector[i]);
            }
            while (endOfSecondVectorNotReached) {
                float value = queryData->getNextValue(pRowId, pointerToVectorElement);
                element.val += value * value;
                ++pointerToVectorElement;
                endOfSecondVectorNotReached = pointerToVectorElement < queryData->getSizeOfInstance(pRowId);
            }
            pointerToMatrixElement = 0;
            pointerToVectorElement = 0;
            // square root of the sum
            // std::cout << "value: " << element.val;
            element.val = sqrt(element.val);
            // std::cout << " : " << element.val << ", ";
            
            returnValue[i] = element;
        }
        // std::cout << std::endl;
        size_t numberOfElementsToSort = pNneighbors;
        if (numberOfElementsToSort > returnValue.size()) {
            numberOfElementsToSort = returnValue.size();
        }
        // sort the values by increasing order
        // std::sort(returnValue.begin(), returnValue.end(), mapSortAscByValueFloat);
        std::partial_sort(returnValue.begin(), returnValue.begin()+numberOfElementsToSort, returnValue.end(), mapSortAscByValueFloat);
        return returnValue;
    };

    std::vector<sortMapFloat> cosineSimilarity(const std::vector<int> pRowIdVector, const size_t pNneighbors, 
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
            element.val = 0.0;
            float dotProduct = 0.0;
            float magnitudeFirstVector = 0.0;
            float magnitudeSecondVector = 0.0;

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
            element.val = dotProduct / static_cast<float>(magnitudeFirstVector * magnitudeSecondVector);
            
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
