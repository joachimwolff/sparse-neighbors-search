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


static PyObject* createObject(PyObject* self, PyObject* args) {

    size_t numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
    nNeighbors, minimalBlocksInCommon, maxBinSize,
    maximalNumberOfHashCollisions, excessFactor, fast;

    if (!PyArg_ParseTuple(args, "kkkkkkkkk", &numberOfHashFunctions,
                        &blockSize, &numberOfCores, &chunkSize, &nNeighbors,
                        &minimalBlocksInCommon, &maxBinSize,
                        &maximalNumberOfHashCollisions, &excessFactor, &fast))
        return NULL;
    MinHash* minHash = new MinHash (numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
                    maxBinSize, nNeighbors, minimalBlocksInCommon, 
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

static PyObject* fit(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;
    
    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

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

    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    umapVector* instanceFeatureVector = rawData_.inverseIndexData;
    csrMatrix* originalDataMatrix = rawData_.matrixData;

    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    minHash->set_mOriginalData(originalDataMatrix);

    minHash->fit(instanceFeatureVector);

    addressMinHashObject = reinterpret_cast<size_t>(minHash);
    PyObject * pointerToInverseIndex = Py_BuildValue("k", addressMinHashObject);
    
    return pointerToInverseIndex;
}
static PyObject* partialFit(PyObject* self, PyObject* args) {
    return fit(self, args);
}
static PyObject* kneighbors(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t nNeighbors;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;

    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkk", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &nNeighbors, &addressMinHashObject))
        return NULL;

    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);

    // compute the k-nearest neighbors
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, nNeighbors);
   
    size_t sizeOfNeighorList = neighborhood_.neighbors->size();

    PyObject * outerListNeighbors = PyList_New(sizeOfNeighorList);
    PyObject * outerListDistances = PyList_New(sizeOfNeighorList);

    for (size_t i = 0; i < sizeOfNeighorList; ++i) {
        size_t sizeOfInnerNeighborList = neighborhood_.neighbors[i].size();
        PyObject * innerListNeighbors = PyList_New(sizeOfInnerNeighborList);
        PyObject * innerListDistances = PyList_New(sizeOfInnerNeighborList);

        for (size_t j = 0; j < sizeOfInnerNeighborList; ++j) {
            PyObject* valueNeighbor = Py_BuildValue("k", neighborhood_.neighbors->operator[](i)[j]);
            PyList_SetItem(innerListNeighbors, j, valueNeighbor);
            PyObject* valueDistance = Py_BuildValue("f", neighborhood_.distances->operator[](i)[j]);
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
static PyObject* kneighborsGraph(PyObject* self, PyObject* args) {
    PyObject* foo;
    return foo;
}
static PyObject* radiusNeighbors(PyObject* self, PyObject* args) {
PyObject* foo;
    return foo;}
static PyObject* radiusNeighborsGraph(PyObject* self, PyObject* args) {
PyObject* foo;
    return foo;
}
static PyObject* fitKneighbors(PyObject* self, PyObject* args) {
PyObject* foo;
    return foo;
}
static PyObject* fitKneighborsGraph(PyObject* self, PyObject* args) {
PyObject* foo;
    return foo;
}
static PyObject* fitRadiusNeighbors(PyObject* self, PyObject* args) {
PyObject* foo;
    return foo;
}
static PyObject* fitRadiusNeighborsGraph(PyObject* self, PyObject* args) {
    PyObject* foo;
    return foo;
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
init_hashUtility(void)
{
    (void) Py_InitModule("_minHash", minHashFunctions);
}