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

#include "typeDefinitions.h"



SparseMatrixFloat* parseRawData(PyObject * instancesListObj, PyObject * featuresListObj, PyObject * dataListObj,
                                              size_t maxNumberOfInstances, size_t maxNumberOfFeatures) {
    PyObject * instanceSize_tObj;
    PyObject * featureSize_tObj;
    PyObject * dataSize_tObj;

    size_t instanceOld = 0;
    size_t sizeOfFeatureVector = PyList_Size(instancesListObj);

    vsize_t* features = new std::vector<size_t>();
    vfloat* values = new std::vector<float>();
    SparseMatrixFloat* originalData = new SparseMatrixFloat(maxNumberOfInstances);

    size_t instanceCount = 0;

    for (size_t i = 0; i < sizeOfFeatureVector; ++i) {
        instanceSize_tObj = PyList_GetItem(instancesListObj, i);
        featureSize_tObj = PyList_GetItem(featuresListObj, i);
        dataSize_tObj = PyList_GetItem(dataListObj, i);
        size_t featureValue;
        size_t instanceValue;
        float dataValue;

        PyArg_Parse(instanceSize_tObj, "k", &instanceValue);
        PyArg_Parse(featureSize_tObj, "k", &featureValue);
        PyArg_Parse(dataSize_tObj, "f", &dataValue);

        if (instanceOld == instanceValue) {
            // featureIds.push_back(featureValue);
            features->push_back(featureValue);
            values->push_back(dataValue);

            instanceOld = instanceValue;
            if (i == sizeOfFeatureVector-1) {
                originalData->insert(instanceCount, reinterpret_cast<size_t> (features), reinterpret_cast<size_t>(values));
            }
        } else {
            if (instanceOld != MAX_VALUE) {
                originalData->insert(instanceCount, reinterpret_cast<size_t> (features), reinterpret_cast<size_t>(values));
                instanceCount++;
            }
            // featureIds.clear();
            // featureIds.push_back(featureValue);
            values = new vfloat();
            features = new vsize_t();
            features->push_back(featureValue);
            values->push_back(dataValue);
            instanceOld = instanceValue;
        }
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
            if (pNeighborhood->distances->operator[](i)[j] <= pRadius) {
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
                PyList_SetItem(innerListNeighbors, j - pCutFirstValue, valueNeighbor);
                
                PyObject* valueDistance = Py_BuildValue("f", pNeighborhood->distances->operator[](i)[j]);
                PyList_SetItem(innerListDistances, j - pCutFirstValue, valueDistance);
                
            }
        } else {
            for (size_t j = 0 + pCutFirstValue; j < sizeOfInnerNeighborList; ++j) {
                PyObject* valueNeighbor = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[j]);
                PyList_SetItem(innerListNeighbors, j - pCutFirstValue, valueNeighbor); 
                PyObject* valueDistance = Py_BuildValue("f", pNeighborhood->distances->operator[](i)[j]);
                PyList_SetItem(innerListDistances, j -pCutFirstValue, valueDistance); 
            }
            for (size_t j = sizeOfInnerNeighborList; j < pNneighbors + pCutFirstValue; ++j) {
                PyObject* valueNeighbor = Py_BuildValue("i", -1);
                PyList_SetItem(innerListNeighbors, j - pCutFirstValue, valueNeighbor);
                PyObject* valueDistance = Py_BuildValue("f", 0.0);
                PyList_SetItem(innerListDistances, j -pCutFirstValue, valueDistance); 
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

static PyObject* buildGraph(const neighborhood* pNeighborhood, const size_t pNneighbors, const size_t pReturnDistance) {
    size_t sizeOfNeighborList = pNeighborhood->neighbors->size();

    PyObject * rowList = PyList_New(sizeOfNeighborList * pNneighbors);
    PyObject * columnList = PyList_New(sizeOfNeighborList * pNneighbors);
    PyObject * dataList = PyList_New(sizeOfNeighborList * pNneighbors);


    for (size_t i = 0; i < sizeOfNeighborList; ++i) {

        PyObject* root = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[0]);
        size_t sizeOfInnerNeighborList = pNeighborhood->neighbors->operator[](i).size();
        for (size_t j = 1; j < sizeOfInnerNeighborList && j < pNneighbors; ++j) {
            PyObject* node = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[j]);
            PyObject* distance;
            if (pReturnDistance) {
                distance = Py_BuildValue("f", pNeighborhood->distances->operator[](i)[j]);
            } else {
                distance = Py_BuildValue("f", 1.0);
            }
            PyList_SetItem(rowList, i+j-1, root);
            PyList_SetItem(columnList, i+j-1 , node);
            PyList_SetItem(dataList, i+j -1, distance); 
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

static PyObject* radiusNeighborhoodGraph(const neighborhood* pNeighborhood, const size_t pRadius, const size_t pReturnDistance) {
    size_t sizeOfNeighborList = pNeighborhood->neighbors->size();

    PyObject * rowList = PyList_New(0);
    PyObject * columnList = PyList_New(0);
    PyObject * dataList = PyList_New(0);

    for (size_t i = 0; i < sizeOfNeighborList; ++i) {

        PyObject* root = Py_BuildValue("i", pNeighborhood->neighbors->operator[](i)[0]);
        size_t sizeOfInnerNeighborList = pNeighborhood->neighbors->operator[](i).size();
        for (size_t j = 1; j < sizeOfInnerNeighborList; ++j) {
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
    delete pNeighborhood->neighbors;
    delete pNeighborhood->distances;
    delete pNeighborhood;
    
    PyObject* graph = PyList_New(3);
    PyList_SetItem(graph, 0, rowList);
    PyList_SetItem(graph, 1, columnList);
    PyList_SetItem(graph, 2, dataList);
    return graph;
}