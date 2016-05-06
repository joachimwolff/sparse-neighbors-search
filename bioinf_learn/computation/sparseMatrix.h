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
    size_t* mSizesOfInstancesCuda;
    size_t* mSizesOfInstancesSSE;
    
    size_t mMaxNnz;
    size_t mNumberOfInstances;
   
    std::unordered_map<size_t, float> mDotProductPrecomputed;
  public:
    SparseMatrixFloat(size_t pNumberOfInstances, size_t pMaxNnz) {
        // std::cout << "pMaxNNz: " << pMaxNnz << std::endl; 
        
        pMaxNnz = pMaxNnz + 32 - (pMaxNnz % 32);
        // std::cout << "pMaxNNz: " << pMaxNnz << std::endl; 
        mSparseMatrix = new int [pNumberOfInstances * pMaxNnz];
        std::fill_n(mSparseMatrix, pNumberOfInstances * pMaxNnz, MAX_VALUE);
        mSparseMatrixValues = new float [pNumberOfInstances * pMaxNnz]();
        mSizesOfInstances = new size_t [pNumberOfInstances];
        mSizesOfInstancesCuda = new size_t [pNumberOfInstances];
        mSizesOfInstancesSSE = new size_t [pNumberOfInstances];
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
        delete [] mSizesOfInstancesCuda;
        delete [] mSizesOfInstancesSSE;
        
        
    };
    
    void precomputeDotProduct() {
        // std::cout << __LINE__ << std::endl;
        
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
            // std::cout << "instance: " << i << ": value: " << value << std::endl;
            
            mDotProductPrecomputed[i] = (float) value;
            value = 0.0;
            value1 = 0.0;
            value2 = 0.0;
            value3 = 0.0;
            // std::cout << "instance: " << i << ": value: " << mDotProductPrecomputed[i] << std::endl;
        }
        // std::cout << "mDotProduct[0]: " << mDotProductPrecomputed[0] << std::endl;
        // std::cout << __LINE__ << std::endl;
        
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
                value += (double) queryData->getNextValue(pIndex, counterInstance) * (double) this->getNextValue(pIndexNeighbor, counterNeighbor);
                ++counterInstance;
                ++counterNeighbor;
            }
        }
        // if (pIndex % 300 == 0) {
        //     printf("foo: %f\n", (float) value);
        // }
        return (float) value; 
        // printf("%i\n", __LINE__);
        // printf ("instance: %i, neighbor: %i\n", pIndex, pIndexNeighbor);
        // printf("%i\n", __LINE__);
        
        // int* foo2 = this->getSparseMatrixIndexPointer(pIndexNeighbor);
        // printf("foo[0]: %i\n", foo2[0]);
        // printf("%i\n", __LINE__);
        // foo2 = foo2 + 0;
        // printf("foo2 + 0: %i\n", foo2[0]);
        // printf("%i\n", __LINE__);
        // float* foo3 = (float *) (this->getSparseMatrixIndex()+ counterNeighbor+ pIndexNeighbor * mMaxNnz);
        // printf("value foo3: %f\n", foo3[0]);
        // printf("%i\n", __LINE__);
        
        // int sizeInstance = queryData->getSizeOfInstanceSSE(pIndex);
        // int sizeNeighbor = this->getSizeOfInstanceSSE(pIndexNeighbor);
        // __m128 value = {0.0, 0.0, 0.0, 0.0};
        // float* pointerInstanceFeature = (float*) queryData->getSparseMatrixIndexPointer(pIndex);
        // float* pointerNeighborFeature = (float*) this->getSparseMatrixIndexPointer(pIndexNeighbor);
        // float* pointerInstanceValue = queryData->getSparseMatrixValuesPointer(pIndex);
        // float* pointerNeighborValue = this->getSparseMatrixValuesPointer(pIndexNeighbor);
        // while (counterInstance <  sizeInstance && counterNeighbor < sizeNeighbor) {
          
        //     if (queryData->getNextElement(pIndex, counterInstance + 3) < this->getNextElement(pIndexNeighbor, counterNeighbor)) {
        //         counterInstance += 4;
        //         continue;
        //     } else if (this->getNextElement(pIndexNeighbor, counterNeighbor + 3) < queryData->getNextElement(pIndex, counterInstance)) {
        //         counterNeighbor += 4;
        //         continue;
        //     } 
       
        //     __m128 featuresInstance = _mm_loadu_ps(pointerInstanceFeature + counterInstance);
        //     __m128 valueInstance = _mm_loadu_ps(pointerInstanceValue + counterInstance);
       
        //     __m128 featuresNeighbor = _mm_loadu_ps(pointerNeighborFeature+ counterNeighbor);
        //     __m128 valueNeighbor = _mm_loadu_ps(pointerNeighborValue + counterNeighbor);
            
        //     __m128 eq = _mm_cmpeq_ps(_mm_shuffle_ps(featuresInstance, featuresInstance, _MM_SHUFFLE(0,0,0,0)), featuresNeighbor);
		//     __m128 product = _mm_mul_ps(_mm_shuffle_ps(valueInstance, valueInstance, _MM_SHUFFLE(0,0,0,0)), valueNeighbor);
        //     value = _mm_add_ps(value, _mm_and_ps(eq, product));
        // // printf("%i\n", __LINE__);
    
        //     eq = _mm_cmpeq_ps(_mm_shuffle_ps(featuresInstance, featuresInstance, _MM_SHUFFLE(1,1,1,1)), featuresNeighbor);
        //     product = _mm_mul_ps(_mm_shuffle_ps(valueInstance, valueInstance, _MM_SHUFFLE(1,1,1,1)), valueNeighbor);
        //     value = _mm_add_ps(value, _mm_and_ps(eq, product));
    
        //     eq = _mm_cmpeq_ps(_mm_shuffle_ps(featuresInstance, featuresInstance, _MM_SHUFFLE(2,2,2,2)), featuresNeighbor);
        //     product = _mm_mul_ps(_mm_shuffle_ps(valueInstance, valueInstance, _MM_SHUFFLE(2,2,2,2)), valueNeighbor);
        //     value = _mm_add_ps(value, _mm_and_ps(eq, product));
    
        //     eq = _mm_cmpeq_ps(_mm_shuffle_ps(featuresInstance, featuresInstance, _MM_SHUFFLE(3,3,3,3)), featuresNeighbor);
        //     product = _mm_mul_ps(_mm_shuffle_ps(valueInstance, valueInstance, _MM_SHUFFLE(3,3,3,3)), valueNeighbor);
            
            
        //     if (queryData->getNextElement(pIndex, counterInstance + 3) == this->getNextElement(pIndexNeighbor, counterNeighbor+3)) {
        //         counterInstance += 4;
        //         counterNeighbor += 4;
        //     } else if (this->getNextElement(pIndexNeighbor, counterNeighbor + 3) < queryData->getNextElement(pIndex, counterInstance+3)) {
        //         counterNeighbor += 4;
        //     } else { //if (queryData->getNextElement(pIndex, counterInstance + 3) < this->getNextElement(pIndexNeighbor, counterNeighbor)) {
        //         counterInstance += 4;
        //     } // else {
                
        //     // }
        // } 
        // return  value[0] + value[1] + value[2] + value[3];
       
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
    size_t* getSparseMatrixSizeOfInstancesSSE() const {
        return mSizesOfInstancesSSE;
    };
    
    size_t* getSparseMatrixSizeOfInstancesCuda() const {
        return mSizesOfInstancesCuda;
    };
    size_t getMaxNnz() const {
        return mMaxNnz;
    };
    size_t getNumberOfInstances() const {
        return mNumberOfInstances;
    };
    int getNextElement(size_t pInstance, size_t pCounter) const {
        // if (pInstance < mNumberOfInstances && pInstance*mMaxNnz+pCounter < mNumberOfInstances*mMaxNnz) {
        //     if (pCounter < mSizesOfInstances[pInstance]) {
                return mSparseMatrix[pInstance*mMaxNnz+pCounter];
        //     }
        // }
        // return MAX_VALUE;
    };
    float getNextValue(size_t pInstance, size_t pCounter) const {
        // if (pInstance < mNumberOfInstances && pInstance*mMaxNnz+pCounter < mNumberOfInstances*mMaxNnz) {
            // if (pCounter < mSizesOfInstances[pInstance]) {
                return mSparseMatrixValues[pInstance*mMaxNnz+pCounter];
            // }
        // }
        // return MAX_VALUE;
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
    size_t getSizeOfInstanceSSE(size_t pInstance) const {
        if (pInstance < mNumberOfInstances) {
            return mSizesOfInstancesSSE[pInstance];
        }
        return 0;
    };
    void insertElement(size_t pInstanceId, size_t pNnzCount, size_t pFeatureId, float pValue) {
        if (pInstanceId*mMaxNnz + pNnzCount < mNumberOfInstances * mMaxNnz) {
            mSparseMatrix[pInstanceId*mMaxNnz + pNnzCount] = static_cast<int> (pFeatureId);
            mSparseMatrixValues[pInstanceId*mMaxNnz + pNnzCount] = pValue;
            // std::cout << mSparseMatrixValues[pInstanceId*mMaxNnz + pNnzCount] << ", " << pValue*1000 << std::endl;
        }
    };

    void insertToSizesOfInstances(size_t pInstanceId, size_t pSizeOfInstance) {
        if (pInstanceId < mNumberOfInstances) {
            mSizesOfInstances[pInstanceId] = pSizeOfInstance;
            mSizesOfInstancesSSE[pInstanceId] = pSizeOfInstance + 4 - (pSizeOfInstance % 4);
            mSizesOfInstancesCuda[pInstanceId] = pSizeOfInstance + 32 - (pSizeOfInstance % 32);
            
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
                                                const size_t pQueryId, SparseMatrixFloat* pQueryData=NULL)  {
        // std::cout << "cosine, queryId: " << pQueryId << std::endl;
       
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
            //  results[instance] = pDotProducts[instance].y / (sqrtf(pDotProducts[instance].x)* sqrtf(pDotProducts[instance].z));
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