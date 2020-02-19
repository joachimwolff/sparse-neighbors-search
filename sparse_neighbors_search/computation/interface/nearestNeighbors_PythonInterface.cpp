/**
 Copyright 2019 Joachim Wolff

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/
#include <Python.h>

#include "../nearestNeighbors.h"

#include "../parsePythonToCpp.h"

static neighborhood* neighborhoodComputation(size_t pNearestNeighborsAddress, PyObject* pInstancesListObj,
                                                PyObject* pFeaturesListObj, PyObject* pDataListObj,
                                                size_t pMaxNumberOfInstances, size_t pMaxNumberOfFeatures, 
                                                size_t pNneighbors, int pFast, int pSimilarity, float pRadius = -1.0) {
    SparseMatrixFloat* originalDataMatrix = NULL;
    // std::cout << __LINE__ << std::endl;
    if (pMaxNumberOfInstances != 0) {
        originalDataMatrix = parseRawData(pInstancesListObj, pFeaturesListObj, pDataListObj, 
                                                    pMaxNumberOfInstances, pMaxNumberOfFeatures);
    }
    // std::cout << __LINE__ << std::endl;

    NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(pNearestNeighborsAddress);
    // std::cout << __LINE__ << std::endl;

    // compute the k-nearest neighbors
    neighborhood* neighbors_ =  nearestNeighbors->kneighbors(originalDataMatrix, pNneighbors, pFast, pSimilarity, pRadius);
    // std::cout << __LINE__ << std::endl;

    if (originalDataMatrix != NULL) {
        delete originalDataMatrix;    
    } 
    // std::cout << __LINE__ << std::endl;

    return neighbors_;
}

static neighborhood* fitNeighborhoodComputation(size_t pNearestNeighborsAddress, PyObject* pInstancesListObj,
                                                PyObject* pFeaturesListObj,PyObject* pDataListObj,
                                                size_t pMaxNumberOfInstances, size_t pMaxNumberOfFeatures, 
                                                size_t pNneighbors, int pFast, int pSimilarity, float pRadius = -1.0) {
    // std::cout << "fitNeighborhoodComputation" << std::endl;
    SparseMatrixFloat* originalDataMatrix = parseRawData(pInstancesListObj, pFeaturesListObj, pDataListObj, 
                                                    pMaxNumberOfInstances, pMaxNumberOfFeatures);
    // get pointer to the minhash object
    NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(pNearestNeighborsAddress);
    nearestNeighbors->set_mOriginalData(originalDataMatrix);

    nearestNeighbors->fit(originalDataMatrix);
    SparseMatrixFloat* emptyMatrix = NULL;
    neighborhood* neighborhood_ = nearestNeighbors->kneighbors(emptyMatrix, pNneighbors, pFast, pSimilarity, pRadius);
    // delete emptyMatrix;
    // std::cout << "fitNeighborhoodComputation done" << std::endl;

    return neighborhood_;
}

static PyObject* createObject(PyObject* self, PyObject* args) {
    size_t numberOfHashFunctions, shingleSize, numberOfCores, chunkSize,
    nNeighbors, minimalBlocksInCommon, maxBinSize,
    maximalNumberOfHashCollisions, excessFactor, hashAlgorithm,
     blockSize, shingle, removeValueWithLeastSigificantBit, gpu_hash, rangeK_Wta;
    int fast, similarity, pruneInverseIndex, removeHashFunctionWithLessEntriesAs;
    float pruneInverseIndexAfterInstance, cpuGpuLoadBalancing;
    
    if (!PyArg_ParseTuple(args, "kkkkkkkkkiiifikkkkfkk", &numberOfHashFunctions,
                        &shingleSize, &numberOfCores, &chunkSize, &nNeighbors,
                        &minimalBlocksInCommon, &maxBinSize,
                        &maximalNumberOfHashCollisions, &excessFactor, &fast, &similarity,
                        &pruneInverseIndex,&pruneInverseIndexAfterInstance, &removeHashFunctionWithLessEntriesAs,
                        &hashAlgorithm, &blockSize, &shingle, &removeValueWithLeastSigificantBit, 
                        &cpuGpuLoadBalancing, &gpu_hash, &rangeK_Wta))
        return NULL;
    NearestNeighbors* nearestNeighbors;
    nearestNeighbors = new NearestNeighbors (numberOfHashFunctions, shingleSize, numberOfCores, chunkSize,
                        maxBinSize, nNeighbors, minimalBlocksInCommon, 
                        excessFactor, maximalNumberOfHashCollisions, fast, similarity, pruneInverseIndex,
                        pruneInverseIndexAfterInstance, removeHashFunctionWithLessEntriesAs, 
                        hashAlgorithm, blockSize, shingle, removeValueWithLeastSigificantBit,
                        cpuGpuLoadBalancing, gpu_hash, rangeK_Wta);

    size_t adressNearestNeighborsObject = reinterpret_cast<size_t>(nearestNeighbors);
    PyObject* pointerToInverseIndex = Py_BuildValue("k", adressNearestNeighborsObject);
    
    return pointerToInverseIndex;
}
static PyObject* deleteObject(PyObject* self, PyObject* args) {
    size_t addressNearestNeighborsObject;

    if (!PyArg_ParseTuple(args, "k", &addressNearestNeighborsObject))
        return Py_BuildValue("i", 1);;

    NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(addressNearestNeighborsObject);
    delete nearestNeighbors;
    return Py_BuildValue("i", 0);
}

static PyObject* fit(PyObject* self, PyObject* args) {
    // std::cout << "Fitting started" << std::endl;
    size_t addressNearestNeighborsObject, maxNumberOfInstances, maxNumberOfFeatures;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkk", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &addressNearestNeighborsObject))
        return NULL;
    
    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids
    SparseMatrixFloat* originalDataMatrix = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    // std::cout << "parsing done" << std::endl;
    // get pointer to the minhash object
    NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(addressNearestNeighborsObject);
    nearestNeighbors->set_mOriginalData(originalDataMatrix);
    // std::cout << "start to fit" << std::endl;
    nearestNeighbors->fit(originalDataMatrix);
    // std::cout << "fitting done" << std::endl;
    addressNearestNeighborsObject = reinterpret_cast<size_t>(nearestNeighbors);
    PyObject * pointerToInverseIndex = Py_BuildValue("k", addressNearestNeighborsObject);
    // std::cout << "Fitting DONE" << std::endl;
    
    return pointerToInverseIndex;
}
static PyObject* partialFit(PyObject* self, PyObject* args) {
    // std::cout << "partial Fitting started" << std::endl;

    size_t addressNearestNeighborsObject, maxNumberOfInstances, maxNumberOfFeatures;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkk", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &addressNearestNeighborsObject))
        return NULL;
    
    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids
    // std::cout << "retrieve old instance" << std::endl;

    SparseMatrixFloat* originalDataMatrix = parseRawData(instancesListObj, featuresListObj, dataListObj, 
                                                    maxNumberOfInstances, maxNumberOfFeatures);
    // get pointer to the minhash object
    NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(addressNearestNeighborsObject);
    size_t numberOfInstancesOld = nearestNeighbors->getOriginalData()->size();
    // std::cout << "numberOfInstancesOld: " << numberOfInstancesOld << std::endl;
    nearestNeighbors->getOriginalData()->addNewInstancesPartialFit(originalDataMatrix);
    // nearestNeighbors->set_mOriginalData(originalDataMatrix);
    // std::cout << "old information retrieved, start partial fit" << std::endl;

    nearestNeighbors->partialFit(originalDataMatrix, numberOfInstancesOld);
    // std::cout << "parsing done" << std::endl;
    delete originalDataMatrix;
    addressNearestNeighborsObject = reinterpret_cast<size_t>(nearestNeighbors);
    PyObject * pointerToInverseIndex = Py_BuildValue("k", addressNearestNeighborsObject);
    
    return pointerToInverseIndex;
}
static PyObject* kneighbors(PyObject* self, PyObject* args) {
    // std::cout << "kneighbors start" << std::endl;
    
    size_t addressNearestNeighborsObject, nNeighbors, maxNumberOfInstances,
            maxNumberOfFeatures, returnDistance;
    int fast, similarity;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkiik", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &nNeighbors, &returnDistance,
                        &fast, &similarity, &addressNearestNeighborsObject))
        return NULL;

    // compute the k-nearest neighbors
    neighborhood* neighborhood_ = neighborhoodComputation(addressNearestNeighborsObject, instancesListObj, featuresListObj, dataListObj, 
                                                maxNumberOfInstances, maxNumberOfFeatures, nNeighbors, fast, similarity);

    size_t cutFirstValue = 0;
    if (PyList_Size(instancesListObj) == 0) {
        cutFirstValue = 1;
    }
    if (nNeighbors == 0) {
        NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(addressNearestNeighborsObject);
        nNeighbors = nearestNeighbors->getNneighbors();
    }
    // std::cout << "kneighbors successful -> bring neighborhood i shape missing" << std::endl;
    return bringNeighborhoodInShape(neighborhood_, nNeighbors, cutFirstValue, returnDistance);
}

static PyObject* kneighborsGraph(PyObject* self, PyObject* args) {
    // std::cout << "kneighbors graph start" << std::endl;

    size_t addressNearestNeighborsObject, nNeighbors, maxNumberOfInstances,
            maxNumberOfFeatures, returnDistance, symmetric;
    int fast, similarity;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkikik", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &nNeighbors, &returnDistance,
                        &fast, &symmetric, &similarity, &addressNearestNeighborsObject))
        return NULL;
    std::cout << __LINE__ << std::endl;
    // compute the k-nearest neighbors
    neighborhood* neighborhood_ = neighborhoodComputation(addressNearestNeighborsObject, instancesListObj, featuresListObj, dataListObj, 
                                                maxNumberOfInstances, maxNumberOfFeatures, nNeighbors, fast, similarity);
    std::cout << __LINE__ << std::endl;
    
    if (nNeighbors == 0) {
        NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(addressNearestNeighborsObject);
    std::cout << __LINE__ << std::endl;
        
        nNeighbors = nearestNeighbors->getNneighbors();
    }
    // std::cout << "kneighbors graph successful -> build graph missing" << std::endl;
    std::cout << __LINE__ << std::endl;

    return buildGraph(neighborhood_, nNeighbors, returnDistance, symmetric);
}
static PyObject* radiusNeighbors(PyObject* self, PyObject* args) {
    size_t addressNearestNeighborsObject, maxNumberOfInstances,
             maxNumberOfFeatures, returnDistance, similarity;
    int fast;
    float radius;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkfkiik", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &radius, &returnDistance,
                        &fast,&similarity, &addressNearestNeighborsObject))
        return NULL;
    // compute the k-nearest neighbors
    neighborhood* neighborhood_ = neighborhoodComputation(addressNearestNeighborsObject, instancesListObj, featuresListObj, dataListObj, 
                                                maxNumberOfInstances, maxNumberOfFeatures, MAX_VALUE, fast, similarity, radius);
    size_t cutFirstValue = 0;
    if (PyList_Size(instancesListObj) == 0) {
        cutFirstValue = 1;
    }
    return radiusNeighborhood(neighborhood_, radius, cutFirstValue, returnDistance); 
}
static PyObject* radiusNeighborsGraph(PyObject* self, PyObject* args) {
    size_t addressNearestNeighborsObject, maxNumberOfInstances,
            maxNumberOfFeatures, returnDistance, symmetric;
    int fast, similarity;
    float radius;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkfkikik", 
                        &PyList_Type, &instancesListObj,
                        &PyList_Type, &featuresListObj,  
                        &PyList_Type, &dataListObj,
                        &maxNumberOfInstances,
                        &maxNumberOfFeatures,
                        &radius, &returnDistance,
                        &fast, &symmetric, &similarity, &addressNearestNeighborsObject))
        return NULL;
    // compute the k-nearest neighbors
    neighborhood* neighborhood_ = neighborhoodComputation(addressNearestNeighborsObject, instancesListObj, featuresListObj, dataListObj, 
                                                maxNumberOfInstances, maxNumberOfFeatures, MAX_VALUE, fast, similarity, radius);
    return radiusNeighborhoodGraph(neighborhood_, radius, returnDistance, symmetric); 
}
static PyObject* fitKneighbors(PyObject* self, PyObject* args) {
    size_t addressNearestNeighborsObject, maxNumberOfInstances, maxNumberOfFeatures,
            nNeighbors, returnDistance;
    int fast, similarity;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkkiik", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &nNeighbors,
                            &returnDistance, &fast,
                            &similarity,
                            &addressNearestNeighborsObject))
        return NULL;

    neighborhood* neighborhood_ = fitNeighborhoodComputation(addressNearestNeighborsObject, instancesListObj, featuresListObj, dataListObj, 
                                                   maxNumberOfInstances, maxNumberOfFeatures, nNeighbors, fast, similarity);
    size_t cutFirstValue = 1;
    if (nNeighbors == 0) {
        NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(addressNearestNeighborsObject);
        nNeighbors = nearestNeighbors->getNneighbors();
    }
    return bringNeighborhoodInShape(neighborhood_, nNeighbors, cutFirstValue, returnDistance);
}
static PyObject* fitKneighborsGraph(PyObject* self, PyObject* args) {
    size_t addressNearestNeighborsObject, maxNumberOfInstances, maxNumberOfFeatures,
            nNeighbors, returnDistance, symmetric;
    int fast, similarity;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkkikik", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &nNeighbors,
                            &returnDistance, &fast, &symmetric,
                            &similarity,
                            &addressNearestNeighborsObject))
        return NULL;

    neighborhood* neighborhood_ = fitNeighborhoodComputation(addressNearestNeighborsObject, instancesListObj, featuresListObj, dataListObj, 
                                                   maxNumberOfInstances, maxNumberOfFeatures, nNeighbors, fast, similarity);
    if (nNeighbors == 0) {
        NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(addressNearestNeighborsObject);
        nNeighbors = nearestNeighbors->getNneighbors();
    }
    return buildGraph(neighborhood_, nNeighbors, returnDistance, symmetric);

}
static PyObject* fitRadiusNeighbors(PyObject* self, PyObject* args) {
    size_t addressNearestNeighborsObject, maxNumberOfInstances, maxNumberOfFeatures,
            returnDistance;
    int fast, similarity;
    float radius;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkfkiik", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &radius, &returnDistance,
                            &fast, &similarity,
                            &addressNearestNeighborsObject))
        return NULL;

    neighborhood* neighborhood_ = fitNeighborhoodComputation(addressNearestNeighborsObject, instancesListObj, featuresListObj, dataListObj, 
                                                   maxNumberOfInstances, maxNumberOfFeatures, MAX_VALUE, fast, similarity, radius); 
    size_t cutFirstValue = 1;
    return radiusNeighborhood(neighborhood_, radius, cutFirstValue, returnDistance); 
}
static PyObject* fitRadiusNeighborsGraph(PyObject* self, PyObject* args) {
    size_t addressNearestNeighborsObject, maxNumberOfInstances, maxNumberOfFeatures,
            returnDistance, symmetric;
    int fast, similarity;
    float radius;
    PyObject* instancesListObj, *featuresListObj, *dataListObj;

    if (!PyArg_ParseTuple(args, "O!O!O!kkfkikik", 
                            &PyList_Type, &instancesListObj, 
                            &PyList_Type, &featuresListObj,
                            &PyList_Type, &dataListObj,
                            &maxNumberOfInstances,
                            &maxNumberOfFeatures,
                            &radius, &returnDistance, &fast,
                            &symmetric, &similarity,
                            &addressNearestNeighborsObject))
        return NULL;

    neighborhood* neighborhood_ = fitNeighborhoodComputation(addressNearestNeighborsObject, instancesListObj, featuresListObj, dataListObj, 
                                                   maxNumberOfInstances, maxNumberOfFeatures, MAX_VALUE, fast, similarity, radius);
    return radiusNeighborhoodGraph(neighborhood_, radius, returnDistance, symmetric); 
}


static PyObject* getDistributionOfInverseIndex(PyObject* self, PyObject* args) {
    size_t addressNearestNeighborsObject;

    if (!PyArg_ParseTuple(args, "k", &addressNearestNeighborsObject))
        return NULL;

    NearestNeighbors* nearestNeighbors = reinterpret_cast<NearestNeighbors* >(addressNearestNeighborsObject);

    distributionInverseIndex* distribution = nearestNeighbors->getDistributionOfInverseIndex();

    return parseDistributionOfInverseIndex(distribution);
}
// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef nearestNeighborsFunctions[] = {
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
    {"create_object", createObject, METH_VARARGS, "Create the c++ object."},
    {"delete_object", deleteObject, METH_VARARGS, "Delete the c++ object by calling the destructor."},
    {"get_distribution_of_inverse_index", getDistributionOfInverseIndex, METH_VARARGS, "Get the distribution of the inverse index."},
    
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef nearestNeighborsModule = {
    PyModuleDef_HEAD_INIT,
    "_nearestNeighbors",
    NULL,
    -1,
    nearestNeighborsFunctions
};
// definition of the module for python
// PyMODINIT_FUNC
// init_nearestNeighbors(void)
// {
//     (void) Py_InitModule("_nearestNeighbors", nearestNeighborsFunctions);
// }

// definition of the module for python
PyMODINIT_FUNC
PyInit__nearestNeighbors(void)
{
    return PyModule_Create(&nearestNeighborsModule);
    // (void) Py_InitModule("_nearestNeighbors", nearestNeighborsModule);
}

// // definition of available functions for python and which function parsing function in c++ should be called.
// static PyMethodDef hicBuildMatrixFunctions[] = {
//     {"buildMatrix", (PyCFunction) buildMatrix, METH_VARARGS, "Builds a Hi-C interaction matrix."},
   
//     {NULL, NULL, 0, NULL}
// };



// // definition of the module for python
// PyMODINIT_FUNC PyInit_hicBuildMatrixCpp(void)
// {
//     return PyModule_Create(&hicBuildMatrixCppModule);
// }