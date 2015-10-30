/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

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
    vsize_t mSparseMatrix;
  public:
    SparseMatrixFloat(size_t rowsCount) {
        mSparseMatrix.resize(rowsCount*2);
    };
    ~SparseMatrixFloat() {
        // for (size_t i = 0; i < mSparseMatrix.size(); i += 2) {
        //     vsize_t* features = reinterpret_cast<vsize_t* >(mSparseMatrix[i]);
        //     delete features;
        // }
        // for (size_t i = 1; i < mSparseMatrix.size(); i += 2) {
        //     vfloat* values = reinterpret_cast<vfloat* >(mSparseMatrix[i]);
        //     delete values;
        // }
    };
    size_t size() const {
        return mSparseMatrix.size() / 2;
    };
    void insert(size_t rowID, size_t column, size_t value) {

        mSparseMatrix[rowID*2] = column; 
        mSparseMatrix[rowID*2 + 1] = value;
    };

    SparseMatrixFloat* getSubMatrixByRowVector(std::vector<int> rowIDs) const {
        SparseMatrixFloat* subMatrix = new SparseMatrixFloat(rowIDs.size());

        for (size_t i = 0; i < rowIDs.size(); ++i) {
            subMatrix->insert(i, mSparseMatrix[rowIDs[i]*2], mSparseMatrix[rowIDs[i]*2 + 1]);
        }
        return subMatrix;
    }    
    vsize_t getRow(size_t rowID) const{
        std::vector<size_t> row(2);
        row[0] = mSparseMatrix[rowID*2];
        row[1] = mSparseMatrix[rowID*2 + 1];
        return row;
    }
    vsize_t* getFeatureRow(size_t rowID) const {
        return reinterpret_cast<vsize_t* >(mSparseMatrix[rowID * 2]);
    }
    std::vector<sortMapFloat>* multiplyVectorAndSort(std::vector<size_t> row) const {

        std::vector<sortMapFloat>* returnValue = new std::vector<sortMapFloat>();
        returnValue->reserve(mSparseMatrix.size());
        vsize_t* features = reinterpret_cast<vsize_t* >(row[0]);
        vfloat* values = reinterpret_cast<vfloat* >(row[1]); 

        for (size_t i = 0; i < mSparseMatrix.size() - 1; i += 2) {
            vsize_t* featuresMatrix = reinterpret_cast<vsize_t* >(mSparseMatrix[i]);
            vfloat* valuesMatrix = reinterpret_cast<vfloat* >(mSparseMatrix[i + 1]); 
            sortMapFloat element; 
            element.key = i / 2;
            element.val = 0.0;
            size_t featurePointer = 0;
            size_t featureMatrixPointer = 0;

            while (featureMatrixPointer < featuresMatrix->size() && featurePointer < features->size()) {
                if ((*featuresMatrix)[featureMatrixPointer] == (*features)[featurePointer]) {
                    element.val += (*valuesMatrix)[featureMatrixPointer] * (*values)[featurePointer];
                    ++featureMatrixPointer;
                    ++featurePointer;
                } else if ((*featuresMatrix)[featureMatrixPointer] < (*features)[featurePointer]) {
                    ++featureMatrixPointer;
                } else {
                    ++featurePointer;
                }
            }
            if (element.val != 0.0) {
                // (*returnValue)[i/2] = element;
                returnValue->push_back(element);
            }
        }
        if (returnValue->size() != 0) {
            std::sort(returnValue->begin(), returnValue->end(), mapSortDescByValueFloat);
            // std::cout << "\ninstance: " << std::endl;
            // for (auto it = returnValue->begin(); it != returnValue->end(); ++it) {
            //     std::cout << "Key: " << it->key << " value: " << it->val << std::endl;
            // }
            // std::cout << std::endl;
        } else {
            // std::cout << 'no values found' << std::endl;
        }
        
        return returnValue;
    };

};
#endif // SPARSE_MATRIX_H
