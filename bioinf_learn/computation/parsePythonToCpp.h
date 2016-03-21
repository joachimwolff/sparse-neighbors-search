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
#include <Python.h>
#include <iostream>
#include <stdio.h>
#include "typeDefinitions.h"


#ifndef PARSE_H
#define PARSE_H

SparseMatrixFloat* parseRawData(PyObject * pInstancesListObj, PyObject * pFeaturesListObj, PyObject * pDataListObj,
                                              size_t pMaxNumberOfInstances, size_t pMaxNumberOfFeatures) {
    PyObject * instanceSize_tObj;
    PyObject * featureSize_tObj;
    PyObject * dataSize_tObj;
    printf("Max number of instances: %i\n", pMaxNumberOfInstances);
    size_t instanceOld = 0;
    size_t sizeOfFeatureVector = PyList_Size(pInstancesListObj);
    SparseMatrixFloat* originalData = new SparseMatrixFloat(pMaxNumberOfInstances, pMaxNumberOfFeatures);
    size_t featuresCount = 0;
    size_t featureValue;
    size_t instanceValue;
    float dataValue;
    for (size_t i = 0; i < sizeOfFeatureVector; ++i) {
        instanceSize_tObj = PyList_GetItem(pInstancesListObj, i);
        featureSize_tObj = PyList_GetItem(pFeaturesListObj, i);
        dataSize_tObj = PyList_GetItem(pDataListObj, i); 
        
        PyArg_Parse(instanceSize_tObj, "k", &instanceValue);
        PyArg_Parse(featureSize_tObj, "k", &featureValue);
        PyArg_Parse(dataSize_tObj, "f", &dataValue);

        if (instanceOld != instanceValue) {
            // printf("size of instance %u: %u\n", instanceOld, featuresCount);
            originalData->insertToSizesOfInstances(instanceOld, featuresCount);
            while (instanceOld+1 != instanceValue) {
                ++instanceOld;
                originalData->insertToSizesOfInstances(instanceOld, 0);
            }
            featuresCount = 0;
        }
        originalData->insertElement(instanceValue, featuresCount, featureValue, dataValue);
        instanceOld = instanceValue;
        ++featuresCount;
    }
    // printf("size of instance %u: %u\n", instanceOld, featuresCount);
    
    originalData->insertToSizesOfInstances(instanceOld, featuresCount);
    while (instanceOld+1 < pMaxNumberOfInstances) {
        ++instanceOld;
        originalData->insertToSizesOfInstances(instanceOld, 0);
    }

    return originalData;
}

static PyObject* radiusNeighborhood(const neighborhood* pNeighborhood, const size_t pRadius, const size_t pCutFirstValue, const size_t pReturnDistance) {
    size_t sizeOfNeighborList = pNeighborhood->neighbors->size();
    PyObject * outerListNeighbors = PyList_New(sizeOfNeighborList);
    PyObject * outerListDistances = PyList_New(sizeOfNeighborList);

    for (size_t i = 0; i < sizeOfNeighborList; ++i) {
        size_t sizeOfInnerNeighborList = pNeighborhood->neighbors->operator[](i).size();
        PyObject* innerListNeighbors = PyList_New(0);
        PyObject* innerListDistances = PyList_New(0);

        for (size_t j = 0 + pCutFirstValue; j < sizeOfInnerNeighborList; ++j) {
            if (pNeighborhood->distances->operator[](i)[j] <= pRadius && pNeighborhood->neighbors->operator[](i)[j] != -1) {
                PyObject* valueNeighbor = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[j]);
                PyList_SetItem(innerListNeighbors, j - pCutFirstValue, valueNeighbor);
                
                PyObject* valueDistance = Py_BuildValue("f", pNeighborhood->distances->operator[](i)[j]);
                PyList_SetItem(innerListDistances, j - pCutFirstValue, valueDistance);
            } else {
                break;
            }
        }
        PyList_SetItem(outerListNeighbors, i, innerListNeighbors);
        PyList_SetItem(outerListDistances, i, innerListDistances);
    }
    delete pNeighborhood->neighbors;
    delete pNeighborhood->distances;
    delete pNeighborhood;
    PyObject * returnList;
    if (pReturnDistance) {
        returnList = PyList_New(2);
        PyList_SetItem(returnList, 0, outerListDistances);
        PyList_SetItem(returnList, 1, outerListNeighbors);
    } else {
        returnList = PyList_New(1);
        PyList_SetItem(returnList, 0, outerListNeighbors);
    }
    return returnList;
}

static PyObject* bringNeighborhoodInShape(const neighborhood* pNeighborhood, const size_t pNneighbors, const size_t pCutFirstValue, const size_t pReturnDistance) {
    size_t sizeOfNeighborList = pNeighborhood->neighbors->size();
    PyObject * outerListNeighbors = PyList_New(sizeOfNeighborList);
    PyObject * outerListDistances = PyList_New(sizeOfNeighborList);

    for (size_t i = 0; i < sizeOfNeighborList; ++i) {
        size_t sizeOfInnerNeighborList = pNeighborhood->neighbors->operator[](i).size();
        PyObject* innerListNeighbors = PyList_New(pNneighbors);
        PyObject* innerListDistances = PyList_New(pNneighbors);
        if (sizeOfInnerNeighborList > pNneighbors) {
            for (size_t j = 0 + pCutFirstValue; j < pNneighbors + pCutFirstValue; ++j) {
                PyObject* valueNeighbor = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[j]);
                if (PyList_SetItem(innerListNeighbors, j - pCutFirstValue, valueNeighbor) == -1) {
                    std::cout << "error: " << __LINE__ << pNeighborhood->neighbors->operator[](i)[j] << std::endl;
                }
                
                PyObject* valueDistance = Py_BuildValue("f", pNeighborhood->distances->operator[](i)[j]);
                if (PyList_SetItem(innerListDistances, j - pCutFirstValue, valueDistance) == -1) {
                    std::cout << "error: " << __LINE__ << pNeighborhood->distances->operator[](i)[j] << std::endl;
                }
                
            }
        } else {
            for (size_t j = 0 + pCutFirstValue; j < sizeOfInnerNeighborList; ++j) {
                PyObject* valueNeighbor = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[j]);
                if (PyList_SetItem(innerListNeighbors, j - pCutFirstValue, valueNeighbor) == -1) {
                    std::cout << "error: " << __LINE__ << pNeighborhood->neighbors->operator[](i)[j] << std::endl;

                }
                PyObject* valueDistance = Py_BuildValue("f", pNeighborhood->distances->operator[](i)[j]);
                if (PyList_SetItem(innerListDistances, j -pCutFirstValue, valueDistance) == -1) {
                    std::cout << "error: " << __LINE__ << pNeighborhood->distances->operator[](i)[j] << std::endl;

                }
            }
            for (size_t j = sizeOfInnerNeighborList; j < pNneighbors + pCutFirstValue; ++j) {
                PyObject* valueNeighbor = Py_BuildValue("i", -1);
                if (PyList_SetItem(innerListNeighbors, j - pCutFirstValue, valueNeighbor) == -1) {
                    std::cout << "error: " << __LINE__ << -1 << std::endl;

                }
                PyObject* valueDistance = Py_BuildValue("f", 0.0);
                if (PyList_SetItem(innerListDistances, j -pCutFirstValue, valueDistance) == -1) {
                    std::cout << "error: " << __LINE__ << 0.0 << std::endl;

                }
            }
        }
        
        if (PyList_SetItem(outerListNeighbors, i, innerListNeighbors) == -1) {
            std::cout << "error setting neighbor list: " << i << std::endl;
        }
        if (PyList_SetItem(outerListDistances, i, innerListDistances) == -1) {
            std::cout << "error setting distnace list: " << i << std::endl;
        }
    }
    delete pNeighborhood->neighbors;
    delete pNeighborhood->distances;
    delete pNeighborhood;

    PyObject * returnList;
    if (pReturnDistance) {
        returnList = PyList_New(2);
        PyList_SetItem(returnList, 0, outerListDistances);
        PyList_SetItem(returnList, 1, outerListNeighbors);
    } else {
        returnList = PyList_New(1);
        PyList_SetItem(returnList, 0, outerListNeighbors); 
    }
    return returnList;
}

static PyObject* buildGraph(const neighborhood* pNeighborhood, const size_t pNneighbors, const size_t pReturnDistance, const size_t symmetric) {
    size_t sizeOfNeighborList = pNeighborhood->neighbors->size();
    PyObject * rowList = PyList_New(0);
    PyObject * columnList = PyList_New(0);
    PyObject * dataList = PyList_New(0);
    std::map<std::pair<size_t, size_t>, float> symmetricMatrix;
    if (symmetric) {
        for (size_t i = 0; i < sizeOfNeighborList; ++i) {
            size_t root = pNeighborhood->neighbors->operator[](i)[0];
            size_t sizeOfInnerNeighborList = pNeighborhood->neighbors->operator[](i).size();

            for (size_t j = 0; j < sizeOfInnerNeighborList && j < pNneighbors; ++j) {
                if (pNeighborhood->neighbors->operator[](i)[j] == -1) break;
                size_t node = pNeighborhood->neighbors->operator[](i)[j];
                float distance;
                if (pReturnDistance) {
                    distance = pNeighborhood->distances->operator[](i)[j];
                } else {
                    distance = 1.0;
                }
                std::pair<size_t, size_t> rootNodePair = std::make_pair(root, node);
                auto it = symmetricMatrix.find(rootNodePair);
                if (it != symmetricMatrix.end()) {
                    symmetricMatrix[rootNodePair] = (symmetricMatrix[rootNodePair] + distance) / 2;
                } else {
                    symmetricMatrix[rootNodePair] = distance;
                }
                rootNodePair = std::make_pair(node, root);
                it = symmetricMatrix.find(rootNodePair);
                if (it != symmetricMatrix.end()) {
                    symmetricMatrix[rootNodePair] = (symmetricMatrix[rootNodePair] + distance) / 2;
                } else {
                    symmetricMatrix[rootNodePair] = distance;
                }
            }
        }

        for (auto itSymmetric = symmetricMatrix.begin(); itSymmetric != symmetricMatrix.end(); ++itSymmetric) {
            PyObject* root = Py_BuildValue("i", itSymmetric->first.first);
            PyObject* node = Py_BuildValue("i", itSymmetric->first.second);
            PyObject* distance = Py_BuildValue("f", itSymmetric->second);
            PyList_Append(rowList, root);
            PyList_Append(columnList, node);
            PyList_Append(dataList, distance);
        }
    } else {

        for (size_t i = 0; i < sizeOfNeighborList; ++i) {

            PyObject* root = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[0]);
            size_t sizeOfInnerNeighborList = pNeighborhood->neighbors->operator[](i).size();
            for (size_t j = 1; j < sizeOfInnerNeighborList && j < pNneighbors; ++j) {
                if (pNeighborhood->neighbors->operator[](i)[j] == -1) break;
                PyObject* node = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[j]);
                PyObject* distance;
                if (pReturnDistance) {
                    distance = Py_BuildValue("f", pNeighborhood->distances->operator[](i)[j]);
                } else {
                    distance = Py_BuildValue("f", 1.0);
                }
                PyList_Append(rowList, root);
                PyList_Append(columnList, node);
                PyList_Append(dataList, distance);
            }
        }
    }
    delete pNeighborhood->neighbors;
    delete pNeighborhood->distances;
    delete pNeighborhood;

    PyObject* graph = PyList_New(3);
    PyList_SetItem(graph, 0, rowList);
    PyList_SetItem(graph, 1, columnList);
    PyList_SetItem(graph, 2, dataList);

    return graph;
}

static PyObject* radiusNeighborhoodGraph(const neighborhood* pNeighborhood, const size_t pRadius, const size_t pReturnDistance, 
                                            const size_t symmetric) {
    size_t sizeOfNeighborList = pNeighborhood->neighbors->size();

    PyObject * rowList = PyList_New(0);
    PyObject * columnList = PyList_New(0);
    PyObject * dataList = PyList_New(0);

    std::map<std::pair<size_t, size_t>, float> symmetricMatrix;
    if (symmetric) {
        for (size_t i = 0; i < sizeOfNeighborList; ++i) {
            size_t root = pNeighborhood->neighbors->operator[](i)[0];
            size_t sizeOfInnerNeighborList = pNeighborhood->neighbors->operator[](i).size();

            for (size_t j = 0; j < sizeOfInnerNeighborList; ++j) {
                if (pNeighborhood->neighbors->operator[](i)[j] == -1) break;
                if (pNeighborhood->distances->operator[](i)[j] <= pRadius) {
                    size_t node = pNeighborhood->neighbors->operator[](i)[j];
                    float distance;
                    if (pReturnDistance) {
                        distance = pNeighborhood->distances->operator[](i)[j];
                    } else {
                        distance = 1.0;
                    }
                    std::pair<size_t, size_t> rootNodePair = std::make_pair(root, node);
                    auto it = symmetricMatrix.find(rootNodePair);
                    if (it != symmetricMatrix.end()) {
                        symmetricMatrix[rootNodePair] = (symmetricMatrix[rootNodePair] + distance) / 2;
                    } else {
                        symmetricMatrix[rootNodePair] = distance;
                    }
                    rootNodePair = std::make_pair(node, root);
                    it = symmetricMatrix.find(rootNodePair);
                    if (it != symmetricMatrix.end()) {
                        symmetricMatrix[rootNodePair] = (symmetricMatrix[rootNodePair] + distance) / 2;
                    } else {
                        symmetricMatrix[rootNodePair] = distance;
                    }
                } else {
                    break;
                }
            }
        }

        for (auto itSymmetric = symmetricMatrix.begin(); itSymmetric != symmetricMatrix.end(); ++itSymmetric) {
            PyObject* root = Py_BuildValue("i", itSymmetric->first.first);
            PyObject* node = Py_BuildValue("i", itSymmetric->first.second);
            PyObject* distance = Py_BuildValue("f", itSymmetric->second);
            PyList_Append(rowList, root);
            PyList_Append(columnList, node);
            PyList_Append(dataList, distance);
        }
    } else {
        for (size_t i = 0; i < sizeOfNeighborList; ++i) {

            PyObject* root = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[0]);
            size_t sizeOfInnerNeighborList = pNeighborhood->neighbors->operator[](i).size();
            for (size_t j = 1; j < sizeOfInnerNeighborList; ++j) {
                if (pNeighborhood->neighbors->operator[](i)[j] == -1) break;
                if (pNeighborhood->distances->operator[](i)[j] <= pRadius) {
                    PyObject* node = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[j]);
                    PyObject* distance;
                    if (pReturnDistance) {
                        distance = Py_BuildValue("f", pNeighborhood->distances->operator[](i)[j]);
                    } else {
                        distance = Py_BuildValue("f", 1.0);
                    }
                    PyList_Append(rowList, root);
                    PyList_Append(columnList, node);
                    PyList_Append(dataList, distance); 
                } else {
                    break;
                }
            }
        }
    }
    delete pNeighborhood->neighbors;
    delete pNeighborhood->distances;
    delete pNeighborhood;
    
    PyObject* graph = PyList_New(3);
    PyList_SetItem(graph, 0, rowList);
    PyList_SetItem(graph, 1, columnList);
    PyList_SetItem(graph, 2, dataList);
    return graph;
}
static PyObject* parseDistributionOfInverseIndex(distributionInverseIndex* distribution) {
    PyObject* distributionVector = PyDict_New();
    for (auto it = distribution->totalCountForOccurenceOfHashValues.begin(); it != distribution->totalCountForOccurenceOfHashValues.end(); ++it) {
        PyDict_SetItem(distributionVector, Py_BuildValue("i", it->first), Py_BuildValue("i", it->second));
    }
    
    PyObject* numberOfCreatedHashValuesPerHashFunctionList = PyList_New(0);
    for (auto it = distribution->numberOfCreatedHashValuesPerHashFunction.begin();
            it != distribution->numberOfCreatedHashValuesPerHashFunction.end(); ++it) {
       PyList_Append(numberOfCreatedHashValuesPerHashFunctionList, Py_BuildValue("i", *it));            
    }
     PyObject* meanForNumberOfValuesPerHashValueList = PyList_New(0);
    for (auto it = distribution->meanForNumberOfValuesPerHashValue.begin();
            it != distribution->meanForNumberOfValuesPerHashValue.end(); ++it) {
       PyList_Append(meanForNumberOfValuesPerHashValueList, Py_BuildValue("i", *it));            
    }
     PyObject* standardDeviationForNumberOfValuesPerHashValueList = PyList_New(0);
    for (auto it = distribution->standardDeviationForNumberOfValuesPerHashValue.begin();
            it != distribution->standardDeviationForNumberOfValuesPerHashValue.end(); ++it) {
       PyList_Append(standardDeviationForNumberOfValuesPerHashValueList, Py_BuildValue("i", *it));            
    }
    PyObject* mean = Py_BuildValue("i", distribution->mean);
    PyObject* standardDeviation = Py_BuildValue("i", distribution->standardDeviation);
    
    PyObject* result = PyList_New(0);
    PyList_Append(result, distributionVector);
    PyList_Append(result, numberOfCreatedHashValuesPerHashFunctionList);
    PyList_Append(result, meanForNumberOfValuesPerHashValueList);
    PyList_Append(result, standardDeviationForNumberOfValuesPerHashValueList);
    PyList_Append(result, mean);
    PyList_Append(result, standardDeviation);
    
    return result;    
}
#endif // PARSE_H