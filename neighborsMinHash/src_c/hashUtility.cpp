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
#include <unordered_map>

#include <string>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <utility>

typedef std::vector<size_t> vsize_t;
typedef std::map< size_t, vsize_t > umapVector;
typedef std::vector<vsize_t > vvsize_t;
typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<std::map<size_t, size_t> > vmSize_tSize_t;

const size_t MAX_VALUE = 2147483647;
const double A = sqrt(2) - 1;
  

class sort_map {
  public:
	size_t key;
	size_t val;
};

bool mapSortDescByValue(const sort_map& a, const sort_map& b) {
    return a.val > b.val;
}

// Return an hash value for a given key in defined range aModulo
 size_t _size_tHashSimple(size_t key, size_t aModulo) {
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return key % aModulo;
}
// compute the signature for one instance given the number of hash functions, the feature ids and a block size
vvsize_t _computeSignature(const size_t numberOfHashFunctions,
                            const umapVector &instanceFeatureVector, const size_t blockSize,
                            const size_t numberOfCores, size_t chunkSize,
                            umapVector* signatureStorage) {

    const size_t sizeOfInstances = instanceFeatureVector.size();
    vvsize_t instanceSignature;
    instanceSignature.resize(sizeOfInstances);
    if (chunkSize <= 0) {
        chunkSize = ceil(instanceFeatureVector.size() / static_cast<float>(numberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for(size_t index = 0; index < instanceFeatureVector.size(); ++index) {

        auto instanceId = instanceFeatureVector.begin();
        std::advance(instanceId, index);
        // compute unique id
        size_t signatureId = 0;
        for (auto itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
                signatureId = _size_tHashSimple((*itFeatures +1) * (signatureId+1) * A, MAX_VALUE);
        }
        auto signatureIt = (*signatureStorage).find(signatureId);
        if (signatureIt != (*signatureStorage).end()) {
#pragma critical
            instanceSignature[index] = signatureIt->second;
            continue;
        }
        // auto itHashValue_InstanceVector;
        vmSize_tSize_t hashStorage;
        vsize_t signatureHash(numberOfHashFunctions);
        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        for(size_t j = 0; j < numberOfHashFunctions; ++j) {
            size_t minHashValue = MAX_VALUE;
            for (auto itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
                size_t hashValue = _size_tHashSimple((*itFeatures +1) * (j+1) * A, MAX_VALUE);
                if (hashValue < minHashValue) {
                    minHashValue = hashValue;
                }
            }
            signatureHash[j] = minHashValue;
        }
        // reduce number of hash values by a factor of blockSize
        size_t k = 0;
        vsize_t signature;
        signature.reserve((numberOfHashFunctions / blockSize) + 1);
        while (k < (numberOfHashFunctions)) {
            // use computed hash value as a seed for the next computation
            size_t signatureBlockValue = signatureHash[k];
            for (size_t j = 0; j < blockSize; ++j) {
                signatureBlockValue = _size_tHashSimple((signatureHash[k+j]) * signatureBlockValue * A, MAX_VALUE);
            }
            signature.push_back(signatureBlockValue);
            k += blockSize; 
        }
#pragma critical

        instanceSignature[index] = signature;
        signatureStorage->operator[](signatureId) = signature;
    }
    return instanceSignature;
}
// compute the complete inverse index for all given instances and theire non null features
std::pair< std::vector<umapVector >*, umapVector* >
    _computeInverseIndex(const size_t numberOfHashFunctions,
        umapVector& instance_featureVector,
        const size_t blockSize, const size_t maxBinSize, const size_t numberOfCores, size_t chunkSize,
        std::vector<umapVector >* inverseIndex,
        umapVector* signatureStorage, const size_t lazyFitting) {

    // if NULL than the inverse index is new created. Otherwise it is extended.
    if (inverseIndex == NULL) {
        inverseIndex = new std::vector<umapVector >();
    }
    if (signatureStorage == NULL) {
        signatureStorage = new umapVector();
    }
    size_t inverseIndexSize = ceil(((float) numberOfHashFunctions / (float) blockSize)+1);
    inverseIndex->resize(inverseIndexSize);
    if (chunkSize <= 0) {
        chunkSize = ceil(instance_featureVector.size() / static_cast<float>(numberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for(size_t index = 0; index < instance_featureVector.size(); ++index){

        auto instanceId = instance_featureVector.begin();
        std::advance(instanceId, index);

        vmSize_tSize_t hashStorage;
        vsize_t signatureHash(numberOfHashFunctions);
        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        size_t signatureId = 0;
        for(size_t j = 0; j < numberOfHashFunctions; ++j) {
            size_t minHashValue = MAX_VALUE;
            for (auto itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
                size_t hashValue = _size_tHashSimple((*itFeatures +1) * (j+1) * A, MAX_VALUE);
                if (hashValue < minHashValue) {
                    minHashValue = hashValue;
                }
                if (!lazyFitting) {
                    if (j == 0) {
                        signatureId = _size_tHashSimple((*itFeatures +1) * (signatureId+1) * A, MAX_VALUE);
                    }
                }
            }
            signatureHash[j] = minHashValue;
        }
        // reduce number of hash values by a factor of blockSize
        size_t k = 0;
        vsize_t signature;
        signature.reserve((numberOfHashFunctions / blockSize) + 1);
        while (k < (numberOfHashFunctions)) {
            // use computed hash value as a seed for the next computation
            size_t signatureBlockValue = signatureHash[k];
            for (size_t j = 0; j < blockSize; ++j) {
                signatureBlockValue = _size_tHashSimple((signatureHash[k+j]) * signatureBlockValue * A, MAX_VALUE);
            }
            signature.push_back(signatureBlockValue);
            k += blockSize; 
        }
        if (!lazyFitting) {
            signatureStorage->operator[](signatureId) = signature;
        } else {
            signatureStorage->operator[](index) = signature;
        }
        // insert in inverse index
#pragma omp critical
        for (size_t j = 0; j < signature.size(); ++j) {
            auto itHashValue_InstanceVector = inverseIndex->operator[](j).find(signature[j]);
            // if for hash function h_i() the given hash values is already stored
            if (itHashValue_InstanceVector != inverseIndex->operator[](j).end()) {
                // insert the instance id if not too many collisions (maxBinSize)
                if (itHashValue_InstanceVector->second.size() < maxBinSize) {
                    // insert only if there wasn't any collisions in the past
                    if (itHashValue_InstanceVector->second.size() > 0 && itHashValue_InstanceVector->second[0] != MAX_VALUE) {
                        itHashValue_InstanceVector->second.push_back(instanceId->first);
                    }
                } else {
                    // too many collisions: delete stored ids and set it to error value -1
                    itHashValue_InstanceVector->second.clear();
                    itHashValue_InstanceVector->second.push_back(MAX_VALUE);
                }
            } else {
                // given hash value for the specific hash function was not avaible: insert new hash value
                vsize_t instanceIdVector;
                instanceIdVector.push_back(instanceId->first);
                inverseIndex->operator[](j)[signature[j]] = instanceIdVector;
            }
        }
    }
    std::pair< std::vector<umapVector >*, umapVector* > returnValue (inverseIndex, signatureStorage);
    return returnValue;
}

std::pair<vvsize_t , vvfloat > _computeNeighbors(
                                const vvsize_t signatures,
                                const size_t sizeOfNeighborhood, const size_t MAX_VALUE,
                                const size_t minimalBlocksInCommon, const size_t maxBinSize,
                                const size_t numberOfCores, const float maximalNumberOfHashCollisions,
                                size_t chunkSize, const size_t excessFactor,
                                const std::vector<umapVector >& inverseIndex) {
#ifdef OPENMP
    omp_set_dynamic(0);
#endif

    std::pair<vvsize_t , vvfloat > returnVector;

    vvsize_t neighbors;
    vvfloat distances;

    neighbors.resize(signatures.size());
    distances.resize(signatures.size());
    if (chunkSize <= 0) {
        chunkSize = ceil(inverseIndex.size() / static_cast<float>(numberOfCores));
    }
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for (size_t i = 0; i < signatures.size(); ++i) {
        std::map<size_t, size_t> neighborhood;
        for (size_t j = 0; j < signatures[i].size(); ++j) {
            size_t hashID = signatures[i][j];
            if (hashID != 0 && hashID != MAX_VALUE) {
                size_t collisionSize = 0;
                auto instances = inverseIndex.at(j).find(hashID);
                if (instances != inverseIndex.at(j).end()) {
                    collisionSize = instances->second.size();
                } else {
                    continue;
                }
                if (collisionSize < maxBinSize && collisionSize > 0) {
                    for (size_t k = 0; k < instances->second.size(); ++k) {
                        neighborhood[instances->second.at(k)] += 1;
                    }
                }
            }
        }
        std::vector< sort_map > neighborhoodVectorForSorting;

        sort_map mapForSorting;
        for (auto it = neighborhood.begin(); it != neighborhood.end(); ++it) {
            mapForSorting.key = (*it).first;
            mapForSorting.val = (*it).second;
            neighborhoodVectorForSorting.push_back(mapForSorting);
        }
        std::sort(neighborhoodVectorForSorting.begin(), neighborhoodVectorForSorting.end(), mapSortDescByValue);
        vsize_t neighborhoodVector;
        std::vector<float> distanceVector;

        size_t sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(sizeOfNeighborhood * excessFactor), neighborhoodVectorForSorting.size());
        size_t count = 0;

        for (auto it = neighborhoodVectorForSorting.begin();
                it != neighborhoodVectorForSorting.end(); ++it) {
            neighborhoodVector.push_back((*it).key);
            distanceVector.push_back(1 - ((*it).val / maximalNumberOfHashCollisions));
            ++count;
            if (count == sizeOfNeighborhoodAdjusted) {
                break;
            }
        }
        neighbors[i] = neighborhoodVector;
        distances[i] = distanceVector;
    }

    returnVector.first = neighbors;
    returnVector.second = distances;
    return returnVector;
}


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
static PyObject* computeSize_tHash(PyObject* self, PyObject* args) {
    size_t key, modulo, seed;

    if (!PyArg_ParseTuple(args, "kkk", &key, &modulo, &seed))
        return NULL;

    return Py_BuildValue("k",   _size_tHashSimple(key * (seed + 1) * A, modulo));
}

// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeInverseIndex(PyObject* self, PyObject* args) {

    size_t numberOfHashFunctions, blockSize, maxBinSize, numberOfCores, chunkSize, lazyFitting;
    PyObject * instancesListObj;
    PyObject * featuresListObj;
   

    if (!PyArg_ParseTuple(args, "kO!O!kkkkk", &numberOfHashFunctions, &PyList_Type, &instancesListObj, &PyList_Type,
                            &featuresListObj, &blockSize, &maxBinSize, &numberOfCores, &chunkSize, &lazyFitting))
        return NULL;

    // parse from python list to a c++ map<size_t, vector<size_t> >
    // where key == instance id and vector<size_t> == non null feature ids
    umapVector instanceFeatureVector = _parseInstancesFeatures(instancesListObj, featuresListObj);

    // compute inverse index in c++
    std::pair<std::vector<umapVector >*, umapVector* > index_signatureStorage =
                     _computeInverseIndex(numberOfHashFunctions, instanceFeatureVector,
                                     blockSize, maxBinSize, numberOfCores, chunkSize, NULL, NULL, lazyFitting);
    std::vector<umapVector >* inverseIndex = index_signatureStorage.first;
    umapVector* signatureStorage = index_signatureStorage.second;
    size_t adressOfInverseIndex = reinterpret_cast<size_t>(inverseIndex);
    size_t adressOfSignatureStorage= reinterpret_cast<size_t>(signatureStorage);

    PyObject * posize_terToInverseIndex = Py_BuildValue("k", adressOfInverseIndex);
    PyObject * posize_terToSignatureStorage = Py_BuildValue("k", adressOfSignatureStorage);
    PyObject * outList = PyList_New(2);
    PyList_SetItem(outList, 0, posize_terToInverseIndex);
    PyList_SetItem(outList, 1, posize_terToSignatureStorage);
    return outList;
}
// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeSignature(PyObject* self, PyObject* args) {

    size_t numberOfHashFunctions, blockSize, numberOfCores, chunkSize;
    size_t addressSignatureStorage;

    PyObject * listInstancesObj;
    PyObject * listFeaturesObj;
    
    if (!PyArg_ParseTuple(args, "kO!O!kkkk", &numberOfHashFunctions,
                        &PyList_Type, &listInstancesObj,
                        &PyList_Type, &listFeaturesObj, &blockSize, &numberOfCores, &chunkSize,
                        &addressSignatureStorage))
        return NULL;

    umapVector instanceFeatureVector = _parseInstancesFeatures(listInstancesObj, listFeaturesObj);


    // get pointer to signature storage
    umapVector* signatureStorage =
            reinterpret_cast<umapVector* >(addressSignatureStorage);
    // compute in c++
    vvsize_t signatures = _computeSignature(numberOfHashFunctions, instanceFeatureVector,
                                                                                    blockSize,numberOfCores, chunkSize,
                                                                                    signatureStorage);
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

    size_t addressInverseIndex;
    size_t addressSignatureStorage;

    size_t numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
    sizeOfNeighborhood, MAX_VALUE, minimalBlocksInCommon, maxBinSize,
    maximalNumberOfHashCollisions, excessFactor, lazyFitting;

    PyObject * listInstancesObj;
    PyObject * listFeaturesObj;
     
    if (!PyArg_ParseTuple(args, "kO!O!kkkkkkkkkkkk", &numberOfHashFunctions,
                        &PyList_Type, &listInstancesObj,
                        &PyList_Type, &listFeaturesObj, &blockSize, 
                        &numberOfCores, &chunkSize, &sizeOfNeighborhood, 
                        &MAX_VALUE, &minimalBlocksInCommon, &maxBinSize,
                        &maximalNumberOfHashCollisions, &excessFactor, 
                        &addressInverseIndex, &addressSignatureStorage, &lazyFitting))
        return NULL;

    umapVector instanceFeatureVector = _parseInstancesFeatures(listInstancesObj, listFeaturesObj);

    // get pointer to inverse index
    std::vector<umapVector >* inverseIndex =
            reinterpret_cast<std::vector<umapVector >* >(addressInverseIndex);
    // get pointer to signature storage
    umapVector* signatureStorage =
            reinterpret_cast<umapVector* >(addressSignatureStorage);

    vvsize_t signatures;
    if (!lazyFitting) {
        // compute signatures of the instances
        signatures = _computeSignature(numberOfHashFunctions, instanceFeatureVector,
                                                                blockSize,numberOfCores, chunkSize, signatureStorage);
    } else {
        // move the instances from map to vector without recomputing it.
        for (umapVector::iterator it = (*signatureStorage).begin(); it != (*signatureStorage).end(); ++it) {
            signatures.push_back(it->second);
        }
    }
    // compute the k-nearest neighbors
    std::pair<vvsize_t , vvfloat > neighborsDistances = 
        _computeNeighbors(signatures, sizeOfNeighborhood, MAX_VALUE, minimalBlocksInCommon, maxBinSize,
                            numberOfCores, maximalNumberOfHashCollisions, chunkSize, excessFactor, *inverseIndex);
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
// parse python call to c++; execute c++ and parse it back to python 
static PyObject* computePartialFit(PyObject* self, PyObject* args)
{
    size_t addressInverseIndex;
    size_t addressSignatureStorage;

    size_t numberOfHashFunctions, blockSize, maxBinSize, numberOfCores, chunkSize;
    PyObject * instancesListObj;
    PyObject * featuresListObj;

    if (!PyArg_ParseTuple(args, "kO!O!kkkkkk", &numberOfHashFunctions, &PyList_Type, &instancesListObj, &PyList_Type,
                            &featuresListObj, &blockSize, &maxBinSize, &numberOfCores, &chunkSize, &addressInverseIndex,
                            &addressSignatureStorage))
        return NULL;
    umapVector instanceFeatureVector = _parseInstancesFeatures(instancesListObj, featuresListObj);

    // get pointer to inverse index
    std::vector<umapVector >* inverseIndex =
            reinterpret_cast<std::vector<umapVector >* >(addressInverseIndex);
    // get pointer to inverse index
    umapVector* signatureStorage =
            reinterpret_cast<umapVector* >(addressSignatureStorage);
    // compute inverse index in c++
    std::pair<std::vector<umapVector >*, umapVector* > index_signatureStorage =
                     _computeInverseIndex(numberOfHashFunctions, instanceFeatureVector,
                                     blockSize, maxBinSize, numberOfCores, chunkSize, inverseIndex, signatureStorage, false);
    inverseIndex = index_signatureStorage.first;
    signatureStorage = index_signatureStorage.second;
    size_t adressOfInverseIndex = reinterpret_cast<size_t>(inverseIndex);
    size_t adressOfSignatureStorage= reinterpret_cast<size_t>(signatureStorage);

    PyObject * pointerToInverseIndex = Py_BuildValue("k", adressOfInverseIndex);
    PyObject * pointerToSignatureStorage = Py_BuildValue("k", adressOfSignatureStorage);
    PyObject * outList = PyList_New(2);
    PyList_SetItem(outList, 0, pointerToInverseIndex);
    PyList_SetItem(outList, 1, pointerToSignatureStorage);
    return outList;
}

// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef size_tHashMethods[] = {
    {"computesize_tHash", computeSize_tHash, METH_VARARGS, "Calculate a hash for given key, modulo and seed."},
    {"computeSignature", computeSignature, METH_VARARGS, "Calculate a signature for a given instance."},
    {"computeInverseIndex", computeInverseIndex, METH_VARARGS, "Calculate the inverse index for the given instances."},
    {"computeNeighborhood", computeNeighborhood, METH_VARARGS, "Calculate the candidate list for the given instances."},
    {"computePartialFit", computePartialFit, METH_VARARGS, "Extend the inverse index with the given instances."},
    {NULL, NULL, 0, NULL}
};

// definition of the module for python
PyMODINIT_FUNC
init_hashUtility(void)
{
    (void) Py_InitModule("_hashUtility", size_tHashMethods);
}
