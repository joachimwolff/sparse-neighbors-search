#include <Python.h>
#include <iostream>

#include "typeDefinitions.h"

rawData parseRawData(PyObject * instancesListObj, PyObject * featuresListObj, PyObject * dataListObj,
                                              size_t maxNumberOfInstances, size_t maxNumberOfFeatures) {
    std::cout << "Parse data start..." << std::endl;
    PyObject * instanceSize_tObj;
    PyObject * featureSize_tObj;
    PyObject * dataSize_tObj;


    umapVector* inverseIndexData = new umapVector();
    // create sparse matrix and inserter
    csrMatrix originalData(maxNumberOfInstances, maxNumberOfFeatures);
    mtl::mat::inserter< csrMatrix >* insertElements = new mtl::mat::inserter< csrMatrix >(originalData);

    vsize_t featureIds;
    size_t instanceOld = 0;
    size_t sizeOfFeatureVector = PyList_Size(instancesListObj);

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
            featureIds.push_back(featureValue);
            (*insertElements)[instanceValue][featureValue] << dataValue;
            instanceOld = instanceValue;
            if (i == sizeOfFeatureVector-1) {
                (*inverseIndexData)[instanceValue] = featureIds;
            }
        } else {
            if (instanceOld != MAX_VALUE) {
                (*inverseIndexData)[instanceOld] = featureIds;
            }
            featureIds.clear();
            featureIds.push_back(featureValue);
            (*insertElements)[instanceValue][featureValue] << dataValue;

            instanceOld = instanceValue;
        }
    }
    // delete inserter to get sparse matrix accessible
    delete insertElements;

    rawData returnValues;
    returnValues.matrixData = &originalData;
    returnValues.inverseIndexData = inverseIndexData;
    std::cout << "Parse data start...Done!" << std::endl;

    return returnValues;
}

static PyObject* bringNeighborhoodInShape(const neighborhood pNeighborhood, const size_t pNneighbors, const size_t cutFirstValue) {
    size_t sizeOfNeighborList = pNeighborhood.neighbors->size();
    std::cout << "Size of query: " << sizeOfNeighborList << std::endl;
    PyObject * outerListNeighbors = PyList_New(sizeOfNeighborList);
    PyObject * outerListDistances = PyList_New(sizeOfNeighborList);

    for (size_t i = 0; i < sizeOfNeighborList; ++i) {
        size_t sizeOfInnerNeighborList = pNeighborhood.neighbors->operator[](i).size();
        PyObject* innerListNeighbors = PyList_New(pNneighbors);
        PyObject* innerListDistances = PyList_New(pNneighbors);
        // std::cout << "Size of pNneighbors; " << pNneighbors << std::endl;
        // std::cout << "sizeOfInnerNeighborList" << sizeOfInnerNeighborList << std::endl;
        // std::cout << "Cut first value: " << cutFirstValue << std::endl;
        if (sizeOfInnerNeighborList > pNneighbors) {
            for (size_t j = 0 + cutFirstValue; j < pNneighbors + cutFirstValue; ++j) {
                PyObject* valueNeighbor = Py_BuildValue("i", pNeighborhood.neighbors->operator[](i)[j]);
                PyList_SetItem(innerListNeighbors, j, valueNeighbor);
                PyObject* valueDistance = Py_BuildValue("f", pNeighborhood.distances->operator[](i)[j]);
                PyList_SetItem(innerListDistances, j, valueDistance);
            }
        } else {
            for (size_t j = 0 + cutFirstValue; j < sizeOfInnerNeighborList; ++j) {
                PyObject* valueNeighbor = Py_BuildValue("i", pNeighborhood.neighbors->operator[](i)[j]);
                PyList_SetItem(innerListNeighbors, j, valueNeighbor);
                PyObject* valueDistance = Py_BuildValue("f", pNeighborhood.distances->operator[](i)[j]);
                PyList_SetItem(innerListDistances, j, valueDistance);
            }
            for (size_t j = sizeOfInnerNeighborList; j < pNneighbors + cutFirstValue; ++j) {
                PyObject* valueNeighbor = Py_BuildValue("i", -1);
                PyList_SetItem(innerListNeighbors, j, valueNeighbor);
                PyObject* valueDistance = Py_BuildValue("f", 0.0);
                PyList_SetItem(innerListDistances, j, valueDistance);
            }
        }
        // std::cout << "Size of list_neighbor: " << PyList_Size(innerListNeighbors) << std::endl;
        // std::cout << "Size of list_distance: " << PyList_Size(innerListDistances) << std::endl;

        PyList_SetItem(outerListNeighbors, i, innerListNeighbors);
        PyList_SetItem(outerListDistances, i, innerListDistances);
    }
    PyObject * returnList = PyList_New(2);
    PyList_SetItem(returnList, 0, outerListDistances);
    PyList_SetItem(returnList, 1, outerListNeighbors);
    std::cout << "Neighbors in interafce coputed!" << std::endl;
    return returnList;
}
