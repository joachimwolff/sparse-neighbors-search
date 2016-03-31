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
#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include "typeDefinitionsBasic.h"

#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H


// struct keyElement {
//     uint32_t x;
//     uint32_t y;
// };
class SparseMatrixFloat {

  private:  
    // stores only the pointer addresses to:
    //      if even index: size_t which is pointer for vsize
    //      if odd indeX: size_t which is pointer for vfloat
    // size_t* mSparseMatrix;
    // float*  mSparseMatrixValues;
    // size_t* mSizesOfInstances;
    size_t mMaxNnz;
    size_t mNumberOfInstances;
    
    std::vector<std::vector<sparseData>* > * mDataMatrix;
   
    std::unordered_map<uint32_t, float> mDotProductPrecomputed;
    // std::vector<std::unordered_map<uint32_t, float>* >* mValuesPrecomputed;
    
    // std::unordered_map<size_t, std::unordered_map<size_t, float>* >* mComputedValues;
    std::vector<std::vector<sparseData>* > * getDataMatrix() {
        return mDataMatrix; 
     };
    
    float dotProduct(const size_t pIndex, const size_t pIndexNeighbor, SparseMatrixFloat* pQueryData=NULL)  {
        SparseMatrixFloat* queryData = this;
         if (pQueryData != NULL) {
            queryData = pQueryData;
        }
        float value = 0.0;  
        std::vector<sparseData>* instance = queryData->getDataMatrix()->operator[](pIndex);
        std::vector<sparseData>* neighbor = this->getDataMatrix()->operator[](pIndexNeighbor);
       
        auto iteratorInstance = instance->begin();
        auto iteratorNeighbor = neighbor->begin();
        while (iteratorInstance != instance->end() && iteratorNeighbor != neighbor->end()) {
            if (iteratorInstance->instance == iteratorNeighbor->instance) {
                value += iteratorInstance->value * iteratorNeighbor->value;
                ++iteratorInstance;
                ++iteratorNeighbor;
            } else if (iteratorInstance->instance < iteratorNeighbor->instance) {
                ++iteratorInstance;
            } else {
                ++iteratorNeighbor;
            }
        }
        
        return value;  
   };
    
    // std::vector<std::unordered_map<uint32_t, float>* >* getValuesPrecomputed() {
    //     return mValuesPrecomputed;
    // };
  public:
    SparseMatrixFloat(size_t pNumberOfInstances, size_t pMaxNnz) {
        std::cout << __LINE__ << std::endl;
        mDataMatrix = new std::vector<std::vector<sparseData>* >(pNumberOfInstances);
        // mValuesPrecomputed = new std::vector<std::unordered_map<uint32_t, float>* >(pNumberOfInstances);
        for (size_t i = 0; i < pNumberOfInstances; ++i) {
            (*mDataMatrix)[i] = new std::vector<sparseData>();
            // (*mValuesPrecomputed)[i] = new std::unordered_map<uint32_t, float>();
        }
        std::cout << __LINE__ << std::endl;
        
    };
    ~SparseMatrixFloat() {
        for (auto it = mDataMatrix->begin(); it != mDataMatrix->end(); ++it) {
            delete (*it);
        }
        delete mDataMatrix;
        // for (auto it = mValuesPrecomputed->begin(); it != mValuesPrecomputed->end(); ++it) {
        //     delete (*it);
        // }
        // delete mValuesPrecomputed;
    };
    size_t getNextElement(size_t pInstance, size_t pIndex)  {
        return (*mDataMatrix)[pInstance]->operator[](pIndex).instance;
        // if (pInstance < mNumberOfInstances && pInstance*mMaxNnz+pCounter < mNumberOfInstances*mMaxNnz) {
        //     if (pCounter < mSizesOfInstances[pInstance]) {
        //         return mSparseMatrix[pInstance*mMaxNnz+pCounter];
        //     }
        // }
        // return MAX_VALUE;
    };
    float getNextValue(size_t pInstance, size_t pIndex)  {
        return (*mDataMatrix)[pInstance]->operator[](pIndex).value;
        
        // if (pInstance < mNumberOfInstances && pInstance*mMaxNnz+pCounter < mNumberOfInstances*mMaxNnz) {
        //     if (pCounter < mSizesOfInstances[pInstance]) {
        //         return mSparseMatrixValues[pInstance*mMaxNnz+pCounter];
        //     }
        // }
        // return MAX_VALUE;
    };
    size_t getMaxNnz() {
        return 10;
    }
    size_t size()  {
        return mDataMatrix->size();
    };
    
    size_t getSizeOfInstance(size_t pInstance)  {
        return (*mDataMatrix)[pInstance]->size();
    };
    
    std::vector<sparseData>* getInstance(size_t pInstance) {
        if (pInstance >= mDataMatrix->size()) return NULL;
        return (*mDataMatrix)[pInstance];
    }
    void insertElement(size_t pInstanceId, size_t pNnzCount, size_t pFeatureId, float pValue) {
        // std::cout << __LINE__ << std::endl;
       
        sparseData data;
        data.instance = static_cast<uint32_t>(pFeatureId);
        data.value = pValue;
        (*(*mDataMatrix)[pInstanceId]).push_back(data);
        // std::cout << __LINE__ << std::endl;
        
    };
    void addNewInstancesPartialFit( SparseMatrixFloat* pMatrix) {
        
    };
     std::unordered_map<uint32_t, float> getDotProductPrecomputed() {
         return mDotProductPrecomputed;
     }
    void precomputeDotProduct() {
        std::cout << __LINE__ << std::endl;
        
        float value = 0.0;
        float value1 = 0.0;
        float value2 = 0.0;
        float value3 = 0.0;
        for (size_t i = 0; i < mDataMatrix->size(); ++i) {
            size_t j = 0;
            for (j = 0; j < (*mDataMatrix)[i]->size(); j += 4) {
                value += pow((*(*mDataMatrix)[i])[j].value, 2);
                value1 += pow((*(*mDataMatrix)[i])[j + 1].value, 2);
                value2 += pow((*(*mDataMatrix)[i])[j + 2].value, 2);
                value3 += pow((*(*mDataMatrix)[i])[j + 3].value, 2);
            }
            while (j < (*mDataMatrix)[i]->size()) {
                value += pow((*(*mDataMatrix)[i])[j].value, 2);
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
        std::cout << __LINE__ << std::endl;
        
    }
    
    std::vector<sortMapFloat> euclidianDistance(const std::vector<int> pRowIdVector, const size_t pNneighbors, 
                                                const size_t pQueryId, SparseMatrixFloat* pQueryData=NULL) {
        
        const uint32_t pRowId = static_cast<uint32_t>(pQueryId);
        
        std::vector<sortMapFloat> returnValue(pRowIdVector.size());
        // std::cout << __LINE__ << std::endl;
        
        const float valueXX = mDotProductPrecomputed[pRowId];
        // std::cout << __LINE__ << std::endl;
        
        float valueXY = 0.0;
        float valueYY = 0.0;
        uint32_t indexOuter = 0;
        uint32_t indexInner = 0;
        uint32_t instance_id;
        // uint32_t neighbor_id = 
        for (size_t i = 0; i < pRowIdVector.size(); ++i) {
            instance_id = static_cast<uint32_t>(pRowIdVector[i]);
            sortMapFloat element; 
            element.key = pRowIdVector[i];
            element.val = 0.0;
            
            if (pRowId < pRowIdVector[i]) {
                indexOuter = static_cast<uint32_t>(pRowId);
                indexInner = static_cast<uint32_t>(pRowIdVector[i]);
            } else {
                indexOuter = static_cast<uint32_t>(pRowIdVector[i]);
                indexInner = static_cast<uint32_t>(pRowId);
            }
        // std::cout << __LINE__ << std::endl;
            
            // auto iteratorEuclidDistance = (*mValuesPrecomputed)[indexOuter]->find(indexInner);
        // std::cout << __LINE__ << std::endl;
            
            // if (iteratorEuclidDistance != (*mValuesPrecomputed)[indexOuter]->end()) {
            //     element.val = (*iteratorEuclidDistance).second;
            // } else {
        // std::cout << __LINE__ << std::endl;
                
                valueXY = this->dotProduct(pRowId, instance_id, pQueryData);
                if (pQueryData == NULL) {
                    valueYY = mDotProductPrecomputed[instance_id];
                } else {
                    valueYY = pQueryData->getDotProductPrecomputed()[instance_id];
                }
                element.val = valueXX - 2* valueXY + valueYY;
                if (element.val <= 0) {
                    element.val = 0;
                } else {
                    element.val = sqrt(valueXX - 2* valueXY + valueYY);
                }
                
                // (*mValuesPrecomputed)[indexOuter]->operator[](indexInner) = element.val;
            // }
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

    std::vector<sortMapFloat> cosineSimilarity(const std::vector<int> pRowIdVector, const size_t pNneighbors, 
                                                const size_t pQueryId, SparseMatrixFloat* pQueryData=NULL) {
        std::cout << "cosine, queryId: " << pQueryId << std::endl;
       
        SparseMatrixFloat* queryData = this;
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
            float dotProduct_ = 0.0;
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
                    dotProduct_ += this->getNextValue(pRowIdVector[i], pointerToMatrixElement) * queryData->getNextValue(pRowId, pointerToVectorElement);
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
            element.val = dotProduct_ / static_cast<float>(magnitudeFirstVector * magnitudeSecondVector);
            
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
