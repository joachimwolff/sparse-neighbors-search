#include <Python.h>

#include "typeDefinitions.h"

rawData parseRawData(PyObject * instancesListObj, PyObject * featuresListObj, PyObject * dataListObj,
                                              size_t maxNumberOfInstances, size_t maxNumberOfFeatures) {
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

    return returnValues;
}