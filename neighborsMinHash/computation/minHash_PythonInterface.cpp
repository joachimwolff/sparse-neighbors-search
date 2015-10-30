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

#include "minHash.h"
#include "parsePythonToCpp.h"

static neighborhood* neighborhoodComputation(size_t pMinHashAddress, PyObject* pInstancesListObj,PyObject* pFeaturesListObj,PyObject* pDataListObj, 
                                                   size_t pMaxNumberOfInstances, size_t pMaxNumberOfFeatures, 
                                                   size_t pNneighbors, int pFast) {
    SparseMatrixFloat* originalDataMatrix = parseRawData(pInstancesListObj, pFeaturesListObj, pDataListObj, 
                                                    pMaxNumberOfInstances, pMaxNumberOfFeatures);
    MinHash* minHash = reinterpret_cast<MinHash* >(pMinHashAddress);
    // compute the k-nearest neighbors
    return minHash->kneighbors(originalDataMatrix, pNneighbors, pFast);
}

static neighborhood* fitNeighborhoodComputation(size_t pMinHashAddress, PyObject* pInstancesListObj,PyObject* pFeaturesListObj,PyObject* pDataListObj, 
                                                   size_t pMaxNumberOfInstances, size_t pMaxNumberOfFeatures, 
                                                   size_t pNneighbors, int pFast) {
    SparseMatrixFloat* originalDataMatrix = parseRawData(pInstancesListObj, pFeaturesListObj, pDataListObj, 
                                                    pMaxNumberOfInstances, pMaxNumberOfFeatures);
    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(pMinHashAddress);
    minHash->set_mOriginalData(originalDataMatrix);

    minHash->fit(originalDataMatrix);
    SparseMatrixFloat* emptyMatrix = new SparseMatrixFloat(0);
    neighborhood* neighborhood_ = minHash->kneighbors(emptyMatrix, pNneighbors, pFast);
    delete emptyMatrix;
    return neighborhood_;
}

static PyObject* createObject(PyObject* self, PyObject* args) {
    size_t numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
    nNeighbors, minimalBlocksInCommon, maxBinSize,
    maximalNumberOfHashCollisions, excessFactor;
    int fast;

    if (!PyArg_ParseTuple(args, "kkkkkkkkki", &numberOfHashFunctions,
                        &blockSize, &numberOfCores, &chunkSize, &nNeighbors,
                        &minimalBlocksInCommon, &maxBinSize,
                        &maximalNumberOfHashCollisions, &excessFactor, &fast))
        return NULL;
    MinHash* minHash = new MinHash (numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
                    maxBinSize, nNeighbors, minimalBlocksInCommon, 
                    excessFactor, maximalNumberOfHashCollisions, fast);
    
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

static PyObject* fit(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, maxNumberOfInstances, maxNumberOfFeatures;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkk", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &addressMinHashObject))
        return NULL;

    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids
    SparseMatrixFloat* originalDataMatrix = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    minHash->set_mOriginalData(originalDataMatrix);
    minHash->fit(originalDataMatrix);
    addressMinHashObject = reinterpret_cast<size_t>(minHash);
    PyObject * pointerToInverseIndex = Py_BuildValue("k", addressMinHashObject);

    return pointerToInverseIndex;
}
static PyObject* partialFit(PyObject* self, PyObject* args) {
    return fit(self, args);
}
static PyObject* kneighbors(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, nNeighbors, maxNumberOfInstances,
            maxNumberOfFeatures, returnDistance;
    int fast;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkik", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &nNeighbors, &returnDistance,
                        &fast, &addressMinHashObject))
        return NULL;
    // compute the k-nearest neighbors
    neighborhood* neighborhood_ = neighborhoodComputation(addressMinHashObject, instancesListObj, featuresListObj, dataListObj, 
                                                maxNumberOfInstances, maxNumberOfFeatures, nNeighbors, fast);

    size_t cutFirstValue = 0;
    if (PyList_Size(instancesListObj) == 0) {
        cutFirstValue = 1;
    }
    if (nNeighbors == 0) {
        MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
        nNeighbors = minHash->getNneighbors();
    }
    return bringNeighborhoodInShape(neighborhood_, nNeighbors, cutFirstValue, returnDistance);
}

static PyObject* kneighborsGraph(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, nNeighbors, maxNumberOfInstances,
            maxNumberOfFeatures, returnDistance;
    int fast;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkik", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &nNeighbors, &returnDistance,
                        &fast, &addressMinHashObject))
        return NULL;
    // compute the k-nearest neighbors
    neighborhood* neighborhood_ = neighborhoodComputation(addressMinHashObject, instancesListObj, featuresListObj, dataListObj, 
                                                maxNumberOfInstances, maxNumberOfFeatures, nNeighbors, fast);
    return buildGraph(neighborhood_, nNeighbors, returnDistance);
}
static PyObject* radiusNeighbors(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, radius, maxNumberOfInstances,
             maxNumberOfFeatures, returnDistance;
    int fast;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkik", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &radius, &returnDistance,
                        &fast, &addressMinHashObject))
        return NULL;
    // compute the k-nearest neighbors
    neighborhood* neighborhood_ = neighborhoodComputation(addressMinHashObject, instancesListObj, featuresListObj, dataListObj, 
                                                maxNumberOfInstances, maxNumberOfFeatures, MAX_VALUE, fast);
    size_t cutFirstValue = 0;
    if (PyList_Size(instancesListObj) == 0) {
        cutFirstValue = 1;
    }
    return radiusNeighborhood(neighborhood_, radius, cutFirstValue, returnDistance); 
}
static PyObject* radiusNeighborsGraph(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, radius, maxNumberOfInstances,
            maxNumberOfFeatures, returnDistance;
    int fast;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkik", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &radius, &returnDistance,
                        &fast, &addressMinHashObject))
        return NULL;
    // compute the k-nearest neighbors
    neighborhood* neighborhood_ = neighborhoodComputation(addressMinHashObject, instancesListObj, featuresListObj, dataListObj, 
                                                maxNumberOfInstances, maxNumberOfFeatures, MAX_VALUE, fast);
    return radiusNeighborhoodGraph(neighborhood_, radius, returnDistance); 
}
static PyObject* fitKneighbors(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, maxNumberOfInstances, maxNumberOfFeatures,
            nNeighbors, returnDistance;
    int fast;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkik", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &nNeighbors,
                            &returnDistance, &fast,
                            &addressMinHashObject))
        return NULL;

    neighborhood* neighborhood_ = fitNeighborhoodComputation(addressMinHashObject, instancesListObj, featuresListObj, dataListObj, 
                                                   maxNumberOfInstances, maxNumberOfFeatures, nNeighbors, fast);
    size_t cutFirstValue = 1;
    if (nNeighbors == 0) {
        MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
        nNeighbors = minHash->getNneighbors();
    }
    return bringNeighborhoodInShape(neighborhood_, nNeighbors, cutFirstValue, returnDistance);
}
static PyObject* fitKneighborsGraph(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, maxNumberOfInstances, maxNumberOfFeatures,
            nNeighbors, returnDistance;
    int fast;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkik", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &nNeighbors,
                            &returnDistance, &fast,
                            &addressMinHashObject))
        return NULL;

    neighborhood* neighborhood_ = fitNeighborhoodComputation(addressMinHashObject, instancesListObj, featuresListObj, dataListObj, 
                                                   maxNumberOfInstances, maxNumberOfFeatures, nNeighbors, fast);
    size_t cutFirstValue = 1;
    if (nNeighbors == 0) {
        MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
        nNeighbors = minHash->getNneighbors();
    }
    return buildGraph(neighborhood_, nNeighbors, returnDistance);

}
static PyObject* fitRadiusNeighbors(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, maxNumberOfInstances, maxNumberOfFeatures,
            radius, returnDistance;
    int fast;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkik", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &radius, &returnDistance,
                            &fast, 
                            &addressMinHashObject))
        return NULL;

    neighborhood* neighborhood_ = fitNeighborhoodComputation(addressMinHashObject, instancesListObj, featuresListObj, dataListObj, 
                                                   maxNumberOfInstances, maxNumberOfFeatures, MAX_VALUE, fast); 
    size_t cutFirstValue = 1;
    return radiusNeighborhood(neighborhood_, radius, cutFirstValue, returnDistance); 
}
static PyObject* fitRadiusNeighborsGraph(PyObject* self, PyObject* args) {
    size_t addressMinHashObject, maxNumberOfInstances, maxNumberOfFeatures,
            radius, returnDistance;
    int fast;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkik", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &radius, &returnDistance, &fast,
                            &addressMinHashObject))
        return NULL;

    neighborhood* neighborhood_ = fitNeighborhoodComputation(addressMinHashObject, instancesListObj, featuresListObj, dataListObj, 
                                                   maxNumberOfInstances, maxNumberOfFeatures, MAX_VALUE, fast);
    return radiusNeighborhoodGraph(neighborhood_, radius, returnDistance); 
}
// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef minHashFunctions[] = {
    {"fit", fit, METH_VARARGS, "Calculate the inverse index for the given instances."},
    {"partial_fit", partialFit, METH_VARARGS, "Extend the inverse index with the given instances."},
    {"kneighbors", kneighbors, METH_VARARGS, "Calculate k-nearest neighbors."},
    {"kneighbors_graph", kneighborsGraph, METH_VARARGS, "Calculate k-nearest neighbors as a graph."},
    {"radius_neighbors", radiusNeighbors, METH_VARARGS, "Calculate the neighbors inside a given radius."},
    {"radius_neighbors_graph", radiusNeighborsGraph, METH_VARARGS, "Calculate the neighbors inside a given radius as a graph."},
    {"fit_kneighbors", fitKneighbors, METH_VARARGS, "Fits and calculates k-nearest neighbors."},
    {"fit_kneighbors_graph", fitKneighborsGraph, METH_VARARGS, "Fits and calculates k-nearest neighbors as a graph."},
    {"fit_radius_neighbors", fitRadiusNeighbors, METH_VARARGS, "Fits and calculates the neighbors inside a given radius."},
    {"fit_radius_neighbors_graph", fitRadiusNeighborsGraph, METH_VARARGS, "Fits and calculates the neighbors inside a given radius as a graph."},
    {"createObject", createObject, METH_VARARGS, "Create the c++ object."},
    {"deleteObject", deleteObject, METH_VARARGS, "Delete the c++ object by calling the destructor."},
    {NULL, NULL, 0, NULL}
};

// definition of the module for python
PyMODINIT_FUNC
init_minHash(void)
{
    (void) Py_InitModule("_minHash", minHashFunctions);
}