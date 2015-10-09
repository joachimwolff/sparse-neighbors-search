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
// #include <math.h>

// #ifdef OPENMP
// #include <omp.h>
// #endif
// #include <vector>
// #include <map>
// #include <unordered_map>

// #include <string>
// #include <iostream>
// #include <iterator>
// #include <algorithm>
// #include <utility>

#include <minHash.h>

typedef std::vector<size_t> vsize_t;
typedef std::map< size_t, vsize_t > umapVector;
typedef std::vector<vsize_t > vvsize_t;
typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<std::map<size_t, size_t> > vmSize_tSize_t;


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

    size_t addressMinHashObject, lazyFitting;
    PyObject * instancesListObj;
    PyObject * featuresListObj;

    if (!PyArg_ParseTuple(args, "O!O!kk", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &lazyFitting, &addressMinHashObject))
        return NULL;

    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids
    umapVector instanceFeatureVector = _parseInstancesFeatures(instancesListObj, featuresListObj);

    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    (*minHash).computeInverseIndex(instanceFeatureVector);
    
    adressMinHashObject = reinterpret_cast<size_t>(minHash);
    PyObject * pointerToInverseIndex = Py_BuildValue("k", adressMinHashObject);
    
    return pointerToInverseIndex;
}
// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeSignature(PyObject* self, PyObject* args) {

    size_t addressMinHashObject;

    PyObject * listInstancesObj;
    PyObject * listFeaturesObj;
    
    if (!PyArg_ParseTuple(args, "O!O!k", 
                        &PyList_Type, &listInstancesObj,
                        &PyList_Type, &listFeaturesObj,
                        &addressMinHashObject))
        return NULL;

    umapVector instanceFeatureVector = _parseInstancesFeatures(listInstancesObj, listFeaturesObj);
    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    // compute in c++
    vvsize_t signatures = (*minHash).computeSignature(instanceFeatureVector);
  
    size_t sizeOfSignature = signatures.size();
    PyObject * outListInstancesObj = PyList_New(sizeOfSignature);
    for (size_t i = 0; i < sizeOfSignature; ++i) {
    size_t sizeOfFeature = signatures[i].size();
        PyObject * outListFeaturesObj = PyList_New(sizeOfFeature);
        for (size_t j = 0; j < sizeOfFeature; ++j) {
            PyObject* value = Py_BuildValue("k", signatures[i][j]);
            PyList_SetItem(outListFeaturesObj, j, value);
        }
        PyList_SetItem(outListInstancesObj, i, outListFeaturesObj);
    }

    return outListInstancesObj;
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
    vvsize_t signatures;
    if (!lazyFitting) {
        // compute signatures of the instances
        signatures = (*minHash).computeSignature(instanceFeatureVector);
    } else {
        // move the instances from map to vector without recomputing it.
        for (umapVector::iterator it = (*signatureStorage).begin(); it != (*signatureStorage).end(); ++it) {
            signatures.push_back(it->second);
        }
    }
    // compute the k-nearest neighbors
    std::pair<vvsize_t , vvfloat > neighborsDistances = (*minHash).computeNeighborhood(instanceFeatureVector);
        
    size_t sizeOfNeighorList = neighborsDistances.first.size();

    PyObject * outerListNeighbors = PyList_New(sizeOfNeighorList);
    PyObject * outerListDistances = PyList_New(sizeOfNeighorList);

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

    return returnList;
}

static PyObject* createObject(PyObject* self, PyObject* args) {
    size_t adressMinHashObject;

    size_t numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
    sizeOfNeighborhood, MAX_VALUE, minimalBlocksInCommon, maxBinSize,
    maximalNumberOfHashCollisions, excessFactor, lazyFitting;

    if (!PyArg_ParseTuple(args, "kkkkkkkkkk", &numberOfHashFunctions,
                        &blockSize, &numberOfCores, &chunkSize, &sizeOfNeighborhood, 
                        &minimalBlocksInCommon, &maxBinSize,
                        &maximalNumberOfHashCollisions, &excessFactor, 
                        &lazyFitting))
        return NULL;

    MinHash* minHash(numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
                    maxBinSize, lazyFitting, sizeOfNeighborhood, minimalBlocksInCommon, 
                    excessFactor, maximalNumberOfHashCollisions);
    size_t adressMinHashObject = reinterpret_cast<size_t>(minHash);
    PyObject * pointerToInverseIndex = Py_BuildValue("k", adressMinHashObject);
    
    return pointerToInverseIndex;
}
// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef size_tHashMethods[] = {
    {"computeSignature", computeSignature, METH_VARARGS, "Calculate a signature for a given instance."},
    {"computeInverseIndex", computeInverseIndex, METH_VARARGS, "Calculate the inverse index for the given instances."},
    {"computeNeighborhood", computeNeighborhood, METH_VARARGS, "Calculate the candidate list for the given instances."},
    {"createObject", createObject, METH_VARARGS, "Create the c++ object."},
    {NULL, NULL, 0, NULL}
};

// definition of the module for python
PyMODINIT_FUNC
init_hashUtility(void)
{
    (void) Py_InitModule("_hashUtility", size_tHashMethods);
}
