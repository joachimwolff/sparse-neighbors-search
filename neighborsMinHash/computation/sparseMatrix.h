#include <algorithm>
#include <iostream>
#include "typeDefinitions.h"

#ifndef SPARSE_MATRIX
#define SPARSE_MATRIX


struct sortMapFloat {
    size_t key;
    float val;
};

static bool mapSortDescByValueFloat(const sortMapFloat& a, const sortMapFloat& b) {
        return a.val > b.val;
};

class SparseMatrixFloat {

  private: 
    std::vector<size_t> mSparseMatrix;
  public:
    SparseMatrixFloat(size_t rowsCount) {
        mSparseMatrix.resize(rowsCount*2);
    };
    ~SparseMatrixFloat() {
        for (size_t i = 0; i < mSparseMatrix.size(); i += 2) {
            std::vector<size_t>* features = reinterpret_cast<std::vector<size_t>* >(mSparseMatrix[i]);
            delete features;
        }
        for (size_t i = 1; i < mSparseMatrix.size(); i += 2) {
            std::vector<float>* values = reinterpret_cast<std::vector<float>* >(mSparseMatrix[i]);
            delete values;
        }
    };
    size_t getSize(){
        return mSparseMatrix.size();
    }
    void insert(size_t rowID, size_t column, size_t value) {

        mSparseMatrix[rowID*2] = column; 
        mSparseMatrix[rowID*2 + 1] = value;
    };

    SparseMatrixFloat* getSubMatrixByRowVector(std::vector<int> rowIDs) {
        // std::cout << "SM44" << std::endl;
        // std::cout << "Size of input vector: " << rowIDs.size() << std::endl;
        SparseMatrixFloat* subMatrix = new SparseMatrixFloat(rowIDs.size());
        // std::cout << "size of created sparse matrix: " << subMatrix->getSize() << std::endl;

        for (size_t i = 0; i < rowIDs.size(); ++i) {
            subMatrix->insert(i, mSparseMatrix[rowIDs[i]*2], mSparseMatrix[rowIDs[i]*2 + 1]);
        }
        // std::cout << "SM49" << std::endl;
        // std::cout << "size of created sparse matrix: " << subMatrix->getSize() << std::endl;
        return subMatrix;
    }    
    std::vector<size_t> getRow(size_t rowID) {
        std::vector<size_t> row(2);
        row[0] = mSparseMatrix[rowID*2];
        row[1] = mSparseMatrix[rowID*2 + 1];
        return row;
    }
    std::vector<sortMapFloat> multiplyVectorAndSort(std::vector<size_t> row) {
        // y_i = sum a_ij * x_j
        // std::cout << "SM61" << std::endl;

        std::vector<sortMapFloat> returnValue;
        returnValue.resize(mSparseMatrix.size() / 2);
        std::vector<size_t>* features = reinterpret_cast<std::vector<size_t>* >(row[0]);
        std::vector<float>* values = reinterpret_cast<std::vector<float>* >(row[1]); 
        // std::cout << "SM67" << std::endl;

        for (size_t i = 0; i < mSparseMatrix.size() - 1; i += 2) {
            // get pointers back
        // std::cout << "SM71" << std::endl;
        // std::cout << "i: " << i;
        // std::cout << "mSparseMatrix->size(): " << mSparseMatrix.size() << std::endl;
            std::vector<size_t>* featuresMatrix = reinterpret_cast<std::vector<size_t>* >(mSparseMatrix[i]);
            std::vector<float>* valuesMatrix = reinterpret_cast<std::vector<float>* >(mSparseMatrix[i + 1]); 
        // std::cout << "SM77" << std::endl;
            
            // element for
            sortMapFloat element; 
        // std::cout << "SM81" << std::endl;

            element.key = i / 2;
        // std::cout << "SM84" << std::endl;

            element.val = 0.0;
        // std::cout << "SM87" << std::endl;
            
            // compute sum of products
            size_t featurePointer = 0;
        // std::cout << "SM91" << std::endl;

            size_t featureMatrixPointer = 0;
        // std::cout << "SM94" << std::endl;

            while (featureMatrixPointer < featuresMatrix->size() || featurePointer < features->size()) {
                // std::cout << "SM86" << std::endl;
                // std::cout << "featureMatrixPointer: " << featureMatrixPointer;
                // std::cout << " size of featuresMatrix: " << featuresMatrix->size();
                // std::cout << " featurePointer: " << featurePointer;
                // std::cout << " size of features: " << features->size() << std::endl;
                if ((*featuresMatrix)[featureMatrixPointer] == (*features)[featurePointer]) {
                    element.val += (*valuesMatrix)[featureMatrixPointer] * (*valuesMatrix)[featurePointer];
                    ++featureMatrixPointer;
                    ++featurePointer;
                } else if ((*featuresMatrix)[featureMatrixPointer] < (*features)[featurePointer]) {
                    ++featureMatrixPointer;
                } else {
                    ++featurePointer;
                }
            }
        // std::cout << "SM98" << std::endl;

            returnValue[i/2] = element;
        }
        std::sort(returnValue.begin(), returnValue.end(), mapSortDescByValueFloat);
        // std::cout << "SM103" << std::endl;

        return returnValue;
    };

};
#endif // SPARSE_MATRIX
