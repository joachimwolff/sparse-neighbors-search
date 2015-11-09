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
        mMaxNnz = pMaxNnz;
        mNumberOfInstances = pNumberOfInstances;
    };
    ~SparseMatrixFloat() {
        delete mSparseMatrix;
        delete mSparseMatrixValues;
        delete mSizesOfInstances;
    };
    size_t* getFeatureList() const {
        return mSparseMatrix;
    }
    size_t getNextElement(size_t pInstance, size_t pCounter) const {
        if (pCounter < mSizesOfInstances[pInstance]) {
            return mSparseMatrix[pInstance*mMaxNnz+pCounter];
        } else {
            return MAX_VALUE;
        }
    }
    size_t size() const {
        return mNumberOfInstances;
    }
    float* getSparseMatrixValues() const {
        return mSparseMatrixValues;
    }
    size_t getSizeOfInstance(size_t pInstance) const {
        return mSizesOfInstances[pInstance];
    }
    void insertElement(size_t pInstanceId, size_t pNnzCount, size_t pFeatureId, float pValue) {
        mSparseMatrix[pInstanceId*mMaxNnz + pNnzCount] = pFeatureId;
        mSparseMatrixValues[pInstanceId*mMaxNnz + pNnzCount] = pValue; 
    };

    void insertToSizesOfInstances(size_t pInstanceId, size_t pSizeOfInstance) {
        mSizesOfInstances[pInstanceId] = pSizeOfInstance;
    };

    std::vector<sortMapFloat>* euclidianDistance(const std::vector<int> pRowIdVector, const int pRowId, const size_t pNneighbors) const {
        std::vector<sortMapFloat>* returnValue = new std::vector<sortMapFloat>(pRowIdVector.size());
        size_t pointerToMatrixElement = 0;
        size_t pointerToVectorElement = 0;
        // iterate over all candidates
        for (size_t i = 0; i < pRowIdVector.size(); ++i) {
            sortMapFloat element; 
            element.key = pRowIdVector[i];
            element.val = 0.0;
            // features ids are stored in mSparseMatrix. 
            // every instances starts at index indexId*mMaxNnz --> every instance can store maximal mMaxNnz feature ids
            // how many elements per index are stored is stored in mSizesOfInstances[indexID]
            // iterate until both instances have no more feature ids
            while (pointerToMatrixElement < mSizesOfInstances[pRowIdVector[i]] || pointerToVectorElement < mSizesOfInstances[pRowId]) {
                // are the feature ids of the two instances the same?
                if (mSparseMatrix[pRowIdVector[i]*mMaxNnz + pointerToMatrixElement] == mSparseMatrix[pRowId*mMaxNnz + pointerToVectorElement]) {
                    // if (mSparseMatrixValues[pRowIdVector[i]*mMaxNnz + pointerToMatrixElement] > 1 || mSparseMatrixValues[pRowId*mMaxNnz + pointerToVectorElement] > 1) {
                    //     std::cout << "featureID: " << mSparseMatrix[pRowIdVector[i]*mMaxNnz + pointerToMatrixElement];
                    //     std::cout << "value: " << mSparseMatrixValues[pRowIdVector[i]*mMaxNnz + pointerToMatrixElement];
                    //     std::cout << "\nfeatureId_V: " << mSparseMatrix[pRowId*mMaxNnz + pointerToVectorElement];
                    //     std::cout << "featureID_V " << mSparseMatrixValues[pRowId*mMaxNnz + pointerToVectorElement] << std::endl;
                    // }
                    // if they are the same substract the values, compute the square and sum it up
                    element.val += 
                    pow(mSparseMatrixValues[pRowIdVector[i]*mMaxNnz + pointerToMatrixElement] - mSparseMatrixValues[pRowId*mMaxNnz + pointerToVectorElement], 2);
                    // increase both counters to the next element 
                    ++pointerToMatrixElement;
                    ++pointerToVectorElement;
                } else if (mSparseMatrix[pRowIdVector[i]*mMaxNnz + pointerToMatrixElement] < mSparseMatrix[pRowId*mMaxNnz + pointerToVectorElement]) {
                    // if the feature ids are unequal square only the smaller one and add it to the sum
                    element.val += pow(mSparseMatrixValues[pRowIdVector[i]*mMaxNnz + pointerToMatrixElement], 2);
                    ++pointerToMatrixElement;
                } else {
                    element.val += pow(mSparseMatrixValues[pRowId*mMaxNnz + pointerToVectorElement], 2);
                    ++pointerToVectorElement;
                }
            }

            pointerToMatrixElement = 0;
            pointerToVectorElement = 0;
            // square root of the sum
            element.val = sqrt(element.val);
            
            (*returnValue)[i] = element;
            
        }
        size_t numberOfElementsToSort = pNneighbors;
        if (pNneighbors > returnValue->size()) {
            numberOfElementsToSort = returnValue->size();
        }
        // sort the values by increasing order
        std::partial_sort(returnValue->begin(), returnValue->begin()+numberOfElementsToSort, returnValue->end(), mapSortAscByValueFloat);
        return returnValue;
    };
};
#endif // SPARSE_MATRIX_H
