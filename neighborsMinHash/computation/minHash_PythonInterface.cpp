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


static PyObject* createObject(PyObject* self, PyObject* args) {

    size_t numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
    nNeighbors, minimalBlocksInCommon, maxBinSize,
    maximalNumberOfHashCollisions, excessFactor, fast;

    if (!PyArg_ParseTuple(args, "kkkkkkkkkk", &numberOfHashFunctions,
                        &blockSize, &numberOfCores, &chunkSize, &nNeighbors,
                        &minimalBlocksInCommon, &maxBinSize,
                        &maximalNumberOfHashCollisions, &excessFactor, &fast))
        return NULL;
    MinHash* minHash = new MinHash (numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
                    maxBinSize, nNeighbors, minimalBlocksInCommon, 
                    excessFactor, maximalNumberOfHashCollisions);
    
    size_t adressMinHashObject = reinterpret_cast<size_t>(minHash);
    PyObject* pointerToInverseIndex = Py_BuildValue("k", adressMinHashObject);
    std::cout << "Object for minhash created!" <<std::endl;
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
    // csrMatrix* originalDataMatrix = rawData_.matrixData;

    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    // minHash->set_mOriginalData(originalDataMatrix);

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
    size_t returnDistance;
    size_t fast;

    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkkk", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &nNeighbors, &returnDistance,
                        &fast, &addressMinHashObject))
        return NULL;

    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);

    // compute the k-nearest neighbors
    // rawData pRawData, size_t pNneighbors, size_t pReturnDistance, size_t pFast)
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, nNeighbors, fast);
    size_t cutFirstValue = 0;
    if (rawData_.inverseIndexData->size() == 0) {
        cutFirstValue = 1;
    }
    if (nNeighbors == 0) {
        nNeighbors = minHash->getNneighbors();
    }
    return bringNeighborhoodInShape(neighborhood_, nNeighbors, cutFirstValue, returnDistance);
}


static PyObject* kneighborsGraph(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t nNeighbors;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;
    size_t returnDistance;
    size_t fast;

    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkkk", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &nNeighbors, &returnDistance,
                        &fast, &addressMinHashObject))
        return NULL;

    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);

    // compute the k-nearest neighbors
    // rawData pRawData, size_t pNneighbors, size_t pReturnDistance, size_t pFast)
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, nNeighbors, fast);
    return buildGraph(neighborhood_, nNeighbors, returnDistance);
}
static PyObject* radiusNeighbors(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t radius;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;
    size_t returnDistance;
    size_t fast;

    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkkk", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &radius, &returnDistance,
                        &fast, &addressMinHashObject))
        return NULL;

    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);

    // compute the k-nearest neighbors
    // rawData pRawData, size_t pNneighbors, size_t pReturnDistance, size_t pFast)
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, MAX_VALUE, fast);
    size_t cutFirstValue = 0;
    if (rawData_.inverseIndexData->size() == 0) {
        cutFirstValue = 1;
    }
    return radiusNeighborhood(neighborhood_, radius, cutFirstValue, returnDistance); 
}
static PyObject* radiusNeighborsGraph(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t radius;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;
    size_t returnDistance;
    size_t fast;

    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkkk", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &radius, &returnDistance,
                        &fast, &addressMinHashObject))
        return NULL;

    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);

    // compute the k-nearest neighbors
    // rawData pRawData, size_t pNneighbors, size_t pReturnDistance, size_t pFast)
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, MAX_VALUE, fast);
    return radiusNeighborhoodGraph(neighborhood_, radius, returnDistance); 
}
static PyObject* fitKneighbors(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;
    size_t nNeighbors;
    size_t returnDistance;
    size_t fast;
    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkk", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &nNeighbors,
                            &returnDistance, &fast,
                            &addressMinHashObject))
        return NULL;

    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids

    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    umapVector* instanceFeatureVector = rawData_.inverseIndexData;
    // csrMatrix* originalDataMatrix = rawData_.matrixData;

    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    // minHash->set_mOriginalData(originalDataMatrix);

    minHash->fit(instanceFeatureVector);
    delete rawData_.inverseIndexData;
    rawData_.inverseIndexData = new umapVector();
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, nNeighbors, fast);
    delete rawData_.inverseIndexData;
    size_t cutFirstValue = 0;
    if (rawData_.inverseIndexData->size() == 0) {
        cutFirstValue = 1;
    }
    if (nNeighbors == 0) {
        nNeighbors = minHash->getNneighbors();
    }
    return bringNeighborhoodInShape(neighborhood_, nNeighbors, cutFirstValue, returnDistance);
}
static PyObject* fitKneighborsGraph(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;
    size_t nNeighbors;
    size_t returnDistance;
    size_t fast;
    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkk", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &nNeighbors,
                            &returnDistance, &fast,
                            &addressMinHashObject))
        return NULL;

    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids

    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    umapVector* instanceFeatureVector = rawData_.inverseIndexData;
    // csrMatrix* originalDataMatrix = rawData_.matrixData;

    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    // minHash->set_mOriginalData(originalDataMatrix);

    minHash->fit(instanceFeatureVector);
    delete rawData_.inverseIndexData;
    rawData_.inverseIndexData = new umapVector();
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, nNeighbors, fast);
    delete rawData_.inverseIndexData;
    return buildGraph(neighborhood_, nNeighbors, returnDistance);

}
static PyObject* fitRadiusNeighbors(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;
    size_t radius;
    size_t returnDistance;
    size_t fast;
    
    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkk", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &radius, &returnDistance,
                            &fast, 
                            &addressMinHashObject))
        return NULL;

    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids
    rawData rawData_ = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    umapVector* instanceFeatureVector = rawData_.inverseIndexData;
    // csrMatrix* originalDataMatrix = rawData_.matrixData;

    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    // minHash->set_mOriginalData(originalDataMatrix);

    minHash->fit(instanceFeatureVector);

    // compute the k-nearest neighbors
    delete rawData_.inverseIndexData;
    rawData_.inverseIndexData = new umapVector();
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, MAX_VALUE, fast);
    delete rawData_.inverseIndexData;
    size_t cutFirstValue = 0;
    if (rawData_.inverseIndexData->size() == 0) {
        cutFirstValue = 1;
    }
    return radiusNeighborhood(neighborhood_, radius, cutFirstValue, returnDistance); 
}
static PyObject* fitRadiusNeighborsGraph(PyObject* self, PyObject* args) {
    size_t addressMinHashObject;
    size_t maxNumberOfInstances;
    size_t maxNumberOfFeatures;
    size_t radius;
    size_t returnDistance;
    size_t fast;
    
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
    // csrMatrix* originalDataMatrix = rawData_.matrixData;

    // get pointer to the minhash object
    MinHash* minHash = reinterpret_cast<MinHash* >(addressMinHashObject);
    // minHash->set_mOriginalData(originalDataMatrix);

    minHash->fit(instanceFeatureVector);

    // compute the k-nearest neighbors
    // rawData pRawData, size_t pNneighbors, size_t pReturnDistance, size_t pFast)
    delete rawData_.inverseIndexData;
    rawData_.inverseIndexData = new umapVector();
    neighborhood neighborhood_ = minHash->kneighbors(rawData_, MAX_VALUE, fast);
    size_t cutFirstValue = 0;
    if (rawData_.inverseIndexData->size() == 0) {
        cutFirstValue = 1;
    }
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