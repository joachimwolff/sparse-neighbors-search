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
#include <math.h>

#ifdef OPENMP
#include <omp.h>
#endif
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <iterator>
const int MAX_VALUE = 2147483647;
const double A = sqrt(2) - 1;


// Return an hash value for a given key in defined range aModulo
 int _intHashSimple(int key, int aModulo) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return key % aModulo;
}
// compute the signature for one instance given the number of hash functions, the feature ids and a block size
std::vector<int> _computeSignature(const int numberOfHashFunctions, const std::vector<int>& featureVector, const int blockSize) {

    const int sizeOfFeatureVector = featureVector.size();
    std::vector<int> signatureHash(numberOfHashFunctions);
    for(int i = 0; i < numberOfHashFunctions; ++i) {
        int minHashValue = MAX_VALUE;
        for(int j = 0; j < sizeOfFeatureVector; ++j) {
            int hashValue = _intHashSimple((featureVector[j]+1) * (i+1) * A, MAX_VALUE);
            if (hashValue == 0 || hashValue == MAX_VALUE) {
                continue;
            }
            if (hashValue < minHashValue) {
                minHashValue = hashValue;
            }
        }
        signatureHash[i] = minHashValue;
    }
    // reduce number of hash values by a factor of blockSize
    std::vector<int> signature;
    signature.reserve((numberOfHashFunctions / blockSize) + 1);
    int i = 0;
    while (i < (numberOfHashFunctions - blockSize)) {
        // use computed hash value as a seed for the next computation
        int signatureBlockValue = signatureHash[i];
        for (int j = 0; j < blockSize; ++j){
            signatureBlockValue = _intHashSimple((signatureHash[i+j]) * signatureBlockValue * A, MAX_VALUE);
        }
        signature.push_back(signatureBlockValue);
        i += blockSize;
    }
    return signature; 
}
// compute the complete inverse index for all given instances and theire non null features
std::vector<std::map<int, std::vector<int> > >  _computeInverseIndex(const int numberOfHashFunctions,
                                                                std::map<int, std::vector<int> >& instance_featureVector,
                                                                const int blockSize, const int maxBinSize, const int numberOfCores, int chunkSize) {

    std::vector<std::map<int, std::vector<int> > > inverseIndex;
    int inverseIndexSize = ceil(((float) numberOfHashFunctions / (float) blockSize)+1);
    inverseIndex.resize(inverseIndexSize);
    if (chunkSize == -1) {
        int chunkSize = ceil(instance_featureVector.size() / numberOfCores);
    } else if(chunkSize == 0) {
        chunkSize = numberOfCores;
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for(int index = 0; index < instance_featureVector.size(); ++index){

        std::map<int, std::vector<int> >::iterator instanceId = instance_featureVector.begin();
        std::advance(instanceId, index);

        std::map<int, std::vector<int> >::iterator itHashValue_InstanceVector;
        std::vector<std::map<int, int> > hashStorage;
        std::vector<int> signatureHash(numberOfHashFunctions);
        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        for(int j = 0; j < numberOfHashFunctions; ++j) {
            int minHashValue = MAX_VALUE;
            for (std::vector<int>::iterator itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
                int hashValue = _intHashSimple((*itFeatures +1) * (j+1) * A, MAX_VALUE);
                if (hashValue < minHashValue) {
                    minHashValue = hashValue;
                }
            }
            signatureHash[j] = minHashValue;
        }
        // reduce number of hash values by a factor of blockSize
        int k = 0;
        std::vector<int> signature;
        signature.reserve((numberOfHashFunctions / blockSize) + 1);
        while (k < (numberOfHashFunctions)) {
            // use computed hash value as a seed for the next computation
            int signatureBlockValue = signatureHash[k];
            for (int j = 0; j < blockSize; ++j) {
                signatureBlockValue = _intHashSimple((signatureHash[k+j]) * signatureBlockValue * A, MAX_VALUE);
            }
            signature.push_back(signatureBlockValue);
            k += blockSize; 
        }

        // insert in inverse index
#pragma omp critical
        for (int j = 0; j < signature.size(); ++j) {
            itHashValue_InstanceVector = inverseIndex[j].find(signature[j]);
            // if for hash function h_i() the given hash values is already stored
            if (itHashValue_InstanceVector != inverseIndex[j].end()) {
                // insert the instance id if not too many collisions (maxBinSize)
                if (itHashValue_InstanceVector->second.size() < maxBinSize) {
                    // insert only if there wasn't too any collisions in the past
                    if (itHashValue_InstanceVector->second.size() > 0 && itHashValue_InstanceVector->second[0] != -1) {
                        itHashValue_InstanceVector->second.push_back(instanceId->first);
                    }
                } else {
                    // too many collisions: delete stored ids and set it to error value -1
                    itHashValue_InstanceVector->second.clear();
                    itHashValue_InstanceVector->second.push_back(-1);
                }
            } else {
                // given hash value for the specific hash function was not avaible: insert new hash value
                std::vector<int> instanceIdVector;
                instanceIdVector.push_back(instanceId->first);
                inverseIndex[j][signature[j]] = instanceIdVector;
            }
        }
    }
    return inverseIndex;
}

// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeIntHash(PyObject* self, PyObject* args)
{
    int key, modulo, seed;

    if (!PyArg_ParseTuple(args, "iii", &key, &modulo, &seed))
        return NULL;

    return Py_BuildValue("i",   _intHashSimple(key * (seed + 1) * A, modulo));
}

// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeInverseIndex(PyObject* self, PyObject* args)
{
    int numberOfHashFunctions, blockSize, maxBinSize, numberOfCores, chunkSize;
    std::map<int, std::vector<int> > instance_featureVector;
    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * instanceIntObj;
    PyObject * featureIntObj;

    if (!PyArg_ParseTuple(args, "iO!O!iiii", &numberOfHashFunctions, &PyList_Type, &instancesListObj, &PyList_Type,
                            &featuresListObj, &blockSize, &maxBinSize, &numberOfCores, &chunkSize))
        return NULL;
    std::cout << "Number of cors: " << numberOfCores;
    int sizeOfFeatureVector = PyList_Size(instancesListObj);
    if (sizeOfFeatureVector < 0)	return NULL;
    int instanceOld = -1;
    int insert = 0;
    std::vector<int> featureIds;
    // parse from python list to a c++ map<int, vector<int> >
    // where key == instance id and vector<int> == non null feature ids
    for (int i = 0; i < sizeOfFeatureVector; ++i) {
        instanceIntObj = PyList_GetItem(instancesListObj, i);
        featureIntObj = PyList_GetItem(featuresListObj, i);
        int featureValue;
        int instanceValue;

        PyArg_Parse(instanceIntObj, "i", &instanceValue);
        PyArg_Parse(featureIntObj, "i", &featureValue);

        if (instanceOld == instanceValue) {
            featureIds.push_back(featureValue);
            instanceOld = instanceValue;
            if (i == sizeOfFeatureVector-1) {
                instance_featureVector[instanceValue] = featureIds;
            }
        } else {
            if (instanceOld != -1 ) {
                instance_featureVector[instanceOld] = featureIds;
            }
            featureIds.clear();
            featureIds.push_back(featureValue);
            instanceOld = instanceValue;
        }
    }
    // compute inverse index in c++
    std::vector<std::map<int, std::vector<int> > > inverseIndex = _computeInverseIndex(numberOfHashFunctions, instance_featureVector,
                                     blockSize, maxBinSize, numberOfCores, chunkSize);

    int sizeOfInverseIndex = inverseIndex.size();
    PyObject * outListObj = PyList_New(sizeOfInverseIndex);
    // parse c++ map<int, vector<int> > to python list[dict{hash_value: [featureIds]}]
    // Iterate over every hash function
    for (int i = 0; i < sizeOfInverseIndex; ++i) {
        PyObject* dictionary = PyDict_New();
        // iterate over every hash value
        for (std::map<int, std::vector<int> >::iterator it = inverseIndex[i].begin(); it != inverseIndex[i].end(); ++it) {
            PyObject* list = PyList_New(it->second.size());
            // get all indicies for this hash value
            for (int j = 0; j < it->second.size(); ++j) {
                PyObject* value = Py_BuildValue("i", it->second[j]);
                PyObject* PyListIndexLocal = Py_BuildValue("i", j);
                Py_ssize_t listIndexLocal = PyInt_AsSsize_t(PyListIndexLocal);

                PyList_SetItem(list, listIndexLocal, value);
            }
            PyObject* key = Py_BuildValue("i", it->first);
            PyDict_SetItem(dictionary, key, list);
        }
        PyObject* PyListIndex = Py_BuildValue("i", i);
        Py_ssize_t listIndex = PyInt_AsSsize_t(PyListIndex);

        if (PyList_SetItem(outListObj, listIndex, dictionary) == -1) {
            std::cout << "Failure!" << std::flush;
        }
    }
    return outListObj;
}
// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeSignature(PyObject* self, PyObject* args)
{
    int numberOfHashFunctions, blockSize;
    std::vector<int> featureID;
    PyObject * listObj;
    PyObject * intObj;
  
    if (!PyArg_ParseTuple(args, "iO!i", &numberOfHashFunctions, &PyList_Type, &listObj, &blockSize))
        return NULL;

    int numLines = PyList_Size(listObj);
    if (numLines < 0)	return NULL;

    featureID.reserve(numLines);
    // parse from python list to c++ vector
    for (int i = 0; i < numLines; ++i) {
        intObj = PyList_GetItem(listObj, i);
        int value;
        PyArg_Parse(intObj, "i", &value);
        featureID.push_back(value);
    }
    // compute in c++
    std::vector<int> signature = _computeSignature(numberOfHashFunctions, featureID, blockSize);
    // parse from c++ vector to python list
    int sizeOfSignature = signature.size();
    PyObject * outListObj = PyList_New(sizeOfSignature);
    for (int i = 0; i < sizeOfSignature; ++i) {
    PyObject* value = Py_BuildValue("i", signature[i]);
    PyList_SetItem(outListObj, i, value);
    }
    return outListObj;
}
// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef IntHashMethods[] = {
    {"computeIntHash", computeIntHash, METH_VARARGS, "Calculate a hash for given key, modulo and seed."},
    {"computeSignature", computeSignature, METH_VARARGS, "Calculate a signature for a given instance."},
    {"computeInverseIndex", computeInverseIndex, METH_VARARGS, "Calculate the inverse index for the given instances."},

    {NULL, NULL, 0, NULL}
};

// definition of the module for python
PyMODINIT_FUNC
init_hashUtility(void)
{
    (void) Py_InitModule("_hashUtility", IntHashMethods);
}
