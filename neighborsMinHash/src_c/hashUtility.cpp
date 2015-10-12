/**
 Copyright 2015 Joachim Wolff
 Master Project
 Tutors: Milad Miladi, Fabrizio Costa
 Summer semester 2015

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/
#include <Python.h>

#include <iostream>

#include "minHash.h"

umapVector _parseInstancesFeatures(PyObject * instancesListObj, PyObject * featuresListObj) {
    PyObject * instanceSize_tObj;
    PyObject * featureSize_tObj;
    umapVector instance_featureVector;
    vsize_t featureIds;
    size_t instanceOld = 0;
    size_t sizeOfFeatureVector = PyList_Size(instancesListObj);

    for (size_t i = 0; i < sizeOfFeatureVector; ++i) {
        instanceSize_tObj = PyList_GetItem(instancesListObj, i);
        featureSize_tObj = PyList_GetItem(featuresListObj, i);
        size_t featureValue;
        size_t instanceValue;

        PyArg_Parse(instanceSize_tObj, "k", &instanceValue);
        PyArg_Parse(featureSize_tObj, "k", &featureValue);

        if (instanceOld == instanceValue) {
            featureIds.push_back(featureValue);
            instanceOld = instanceValue;
            if (i == sizeOfFeatureVector-1) {
                instance_featureVector[instanceValue] = featureIds;
            }
        } else {
            if (instanceOld != MAX_VALUE) {
                instance_featureVector[instanceOld] = featureIds;
            }
            featureIds.clear();
            featureIds.push_back(featureValue);
            instanceOld = instanceValue;
        }
    }
    return instance_featureVector;
}

// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeInverseIndex(PyObject* self, PyObject* args) {

    size_t addressMinHashObject;
    PyObject * instancesListObj;
    PyObject * featuresListObj;

    if (!PyArg_ParseTuple(args, "O!O!k", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &addressMinHashObject))
        return NULL;

    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids
    umapVector instanceFeatureVector = _parseInstancesFeatures(instancesListObj, featuresListObj);

    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);

    (*minHash).computeInverseIndex(instanceFeatureVector);

    addressMinHashObject = reinterpret_cast<size_t>(minHash);
    PyObject * pointerToInverseIndex = Py_BuildValue("k", addressMinHashObject);
    
    return pointerToInverseIndex;
}

// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeNeighborhood(PyObject* self, PyObject* args) {

    size_t addressMinHashObject;
    size_t sizeOfNeighborhood, lazyFitting;

    PyObject * listInstancesObj;
    PyObject * listFeaturesObj;
     
    if (!PyArg_ParseTuple(args, "O!O!kkk", 
                        &PyList_Type, &listInstancesObj,
                        &PyList_Type, &listFeaturesObj,  
                        &sizeOfNeighborhood, &lazyFitting, &addressMinHashObject))
        return NULL;

    umapVector instanceFeatureVector = _parseInstancesFeatures(listInstancesObj, listFeaturesObj);

    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);

    umap_pair_vector* signatures;
    std::pair<vvsize_t , vvfloat > neighborsDistances;
    size_t doubleNeighbors = 0;
    if (!lazyFitting) {
        // compute signatures of the instances
        signatures = (*minHash).computeSignatureMap(instanceFeatureVector);
        doubleNeighbors = minHash->getDoubleElementsQuery();
    } else {
        signatures = minHash->getSignatureStorage();
        doubleNeighbors = minHash->getDoubleElementsStorage();
    }
    // compute the k-nearest neighbors
    neighborsDistances = (*minHash).computeNeighbors(signatures, doubleNeighbors);
    std::cout << "111" << std::endl;
   
    size_t sizeOfNeighorList = neighborsDistances.first.size();

    PyObject * outerListNeighbors = PyList_New(sizeOfNeighorList);
    PyObject * outerListDistances = PyList_New(sizeOfNeighorList);
    std::cout << "117" << std::endl;

    for (size_t i = 0; i < sizeOfNeighorList; ++i) {
        size_t sizeOfInnerNeighborList = neighborsDistances.first[i].size();
        PyObject * innerListNeighbors = PyList_New(sizeOfInnerNeighborList);
        PyObject * innerListDistances = PyList_New(sizeOfInnerNeighborList);

        for (size_t j = 0; j < sizeOfInnerNeighborList; ++j) {
            PyObject* valueNeighbor = Py_BuildValue("k", neighborsDistances.first[i][j]);
            PyList_SetItem(innerListNeighbors, j, valueNeighbor);
            PyObject* valueDistance = Py_BuildValue("f", neighborsDistances.second[i][j]);
            PyList_SetItem(innerListDistances, j, valueDistance);
        }
        PyList_SetItem(outerListNeighbors, i, innerListNeighbors);
        PyList_SetItem(outerListDistances, i, innerListDistances);

    }
    PyObject * returnList = PyList_New(2);
    PyList_SetItem(returnList, 0, outerListDistances);
    PyList_SetItem(returnList, 1, outerListNeighbors);
    std::cout << "137" << std::endl;

    return returnList;
}

static PyObject* createObject(PyObject* self, PyObject* args) {

    size_t numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
    sizeOfNeighborhood, minimalBlocksInCommon, maxBinSize,
    maximalNumberOfHashCollisions, excessFactor;

    if (!PyArg_ParseTuple(args, "kkkkkkkkk", &numberOfHashFunctions,
                        &blockSize, &numberOfCores, &chunkSize, &sizeOfNeighborhood,
                        &minimalBlocksInCommon, &maxBinSize,
                        &maximalNumberOfHashCollisions, &excessFactor))
        return NULL;
    MinHash* minHash = new MinHash (numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
                    maxBinSize, sizeOfNeighborhood, minimalBlocksInCommon, 
                    excessFactor, maximalNumberOfHashCollisions);
    
    size_t adressMinHashObject = reinterpret_cast<size_t>(minHash);
    PyObject* pointerToInverseIndex = Py_BuildValue("k", adressMinHashObject);
    
    return pointerToInverseIndex;
}
static PyObject* deleteObject(PyObject* self, PyObject* args) {

    size_t addressMinHashObject;

    if (!PyArg_ParseTuple(args, "k", &addressMinHashObject))
        return NULL;

    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    delete minHash;
    return NULL;
}
// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef size_tHashMethods[] = {
    {"computeInverseIndex", computeInverseIndex, METH_VARARGS, "Calculate the inverse index for the given instances."},
    {"computeNeighborhood", computeNeighborhood, METH_VARARGS, "Calculate the candidate list for the given instances."},
    {"createObject", createObject, METH_VARARGS, "Create the c++ object."},
    {"deleteObject", deleteObject, METH_VARARGS, "Delete the c++ object by calling the destructor."},
    {NULL, NULL, 0, NULL}
};

// definition of the module for python
PyMODINIT_FUNC
init_hashUtility(void)
{
    (void) Py_InitModule("_hashUtility", size_tHashMethods);
}