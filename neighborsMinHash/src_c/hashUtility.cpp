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

const int MAX_VALUE = 2147483647;
const double A = sqrt(2) - 1;
  

class sort_map {
  public:
	int key;
	int val;
};

bool mapSortDescByValue(const sort_map& a, const sort_map& b) {
    return a.val > b.val;
}
// template<typename K, typename V>
// bool mapSortDescByValueI(const std::pair<K, V> a, const std::pair<K, V> b) {
//     return a.second > b.second;
// }

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
std::vector< std::vector<int> > _computeSignature(const int numberOfHashFunctions,
                            const std::map< int, std::vector<int> > &instanceFeatureVector, const int blockSize,
                            const int numberOfCores, int chunkSize,
                            const std::map<int, std::vector<int> >& signatureStorage) {

    const int sizeOfInstances = instanceFeatureVector.size();
    std::vector< std::vector<int> > instanceSignature;
    instanceSignature.resize(sizeOfInstances);
    if (chunkSize <= 0) {
        chunkSize = ceil(instanceFeatureVector.size() / static_cast<float>(numberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for(int index = 0; index < instanceFeatureVector.size(); ++index) {

        std::map<int, std::vector<int> >::const_iterator instanceId = instanceFeatureVector.begin();
        std::advance(instanceId, index);
        std::map<int, std::vector<int> >::const_iterator signatureIt = signatureStorage.find(instanceId->first);
        if (signatureIt != signatureStorage.end()) {
#pragma critical
            instanceSignature[index] = signatureIt->second;
            continue;
        } 
        std::map<int, std::vector<int> >::iterator itHashValue_InstanceVector;
        std::vector<std::map<int, int> > hashStorage;
        std::vector<int> signatureHash(numberOfHashFunctions);
        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        for(int j = 0; j < numberOfHashFunctions; ++j) {
            int minHashValue = MAX_VALUE;
            for (std::vector<int>::const_iterator itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
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
#pragma critical
        instanceSignature[index] = signature;

    }
    return instanceSignature;
}
// compute the complete inverse index for all given instances and theire non null features
std::pair< std::vector<std::map<int, std::vector<int> > >*, std::map<int, std::vector<int> >* >
    _computeInverseIndex(const int numberOfHashFunctions,
        std::map<int, std::vector<int> >& instance_featureVector,
        const int blockSize, const int maxBinSize, const int numberOfCores, int chunkSize,
        std::vector<std::map<int, std::vector<int> > >* inverseIndex,
        std::map<int, std::vector<int> >* signatureStorage ) {

    // if NULL than the inverse index is new created. Otherwise it is extended.
    if (inverseIndex == NULL) {
        inverseIndex = new std::vector<std::map<int, std::vector<int> > >();
    }
    if (signatureStorage == NULL) {
        signatureStorage = new std::map<int, std::vector<int> >();
    }
    int inverseIndexSize = ceil(((float) numberOfHashFunctions / (float) blockSize)+1);
    inverseIndex->resize(inverseIndexSize);
    if (chunkSize <= 0) {
        chunkSize = ceil(instance_featureVector.size() / static_cast<float>(numberOfCores));
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
        signatureStorage->operator[](instanceId->first) = signature;
        for (int j = 0; j < signature.size(); ++j) {
            itHashValue_InstanceVector = inverseIndex->operator[](j).find(signature[j]);
            // if for hash function h_i() the given hash values is already stored
            if (itHashValue_InstanceVector != inverseIndex->operator[](j).end()) {
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
                inverseIndex->operator[](j)[signature[j]] = instanceIdVector;
            }
        }
    }
    std::pair< std::vector<std::map<int, std::vector<int> > >*, std::map<int, std::vector<int> >* > returnValue (inverseIndex, signatureStorage);
    return returnValue;
}

std::pair<std::vector< std::vector<int> > , std::vector< std::vector<float> > > _computeNeighbors(
                                const std::vector< std::vector<int> > signatures,
                                const int sizeOfNeighborhood, const int MAX_VALUE,
                                const int minimalBlocksInCommon, const int maxBinSize,
                                const int numberOfCores, const float maximalNumberOfHashCollisions,
                                int chunkSize, const int excessFactor,
                                const std::vector<std::map<int, std::vector<int> > >& inverseIndex) {
#ifdef OPENMP
    omp_set_dynamic(0);
#endif

    std::pair<std::vector< std::vector<int> > , std::vector< std::vector<float> > > returnVector;

    std::vector< std::vector<int> > neighbors;
    std::vector< std::vector<float> > distances;

    neighbors.resize(signatures.size());
    distances.resize(signatures.size());
    if (chunkSize <= 0) {
        chunkSize = ceil(inverseIndex.size() / static_cast<float>(numberOfCores));
    }
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for (int i = 0; i < signatures.size(); ++i) {
        std::unordered_map<int, int> neighborhood;
        for (int j = 0; j < signatures[i].size(); ++j) {
            int hashID = signatures[i][j];
            if (hashID != 0 && hashID != MAX_VALUE) {
                int collisionSize = 0;
                std::map<int, std::vector<int> >::const_iterator instances = inverseIndex.at(j).find(hashID);
                if (instances != inverseIndex.at(j).end()) {
                    collisionSize = instances->second.size();
                } else {
                    continue;
                }
                if (collisionSize < maxBinSize && collisionSize > 0) {
                    for (int k = 0; k < instances->second.size(); ++k) {
                        neighborhood[instances->second.at(k)] += 1;
                    }
                }
            }
        }
        std::vector< sort_map > neighborhoodVectorForSorting;

        sort_map mapForSorting;
        for (std::unordered_map<int, int>::iterator it = neighborhood.begin(); it != neighborhood.end(); ++it) {
            mapForSorting.key = (*it).first;
            mapForSorting.val = (*it).second;
            neighborhoodVectorForSorting.push_back(mapForSorting);
        }
        std::sort(neighborhoodVectorForSorting.begin(), neighborhoodVectorForSorting.end(), mapSortDescByValue);
        std::vector<int> neighborhoodVector;
        std::vector<float> distanceVector;

        int sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(sizeOfNeighborhood * excessFactor), neighborhoodVectorForSorting.size());
        int count = 0;

        for (std::vector<sort_map>::iterator it = neighborhoodVectorForSorting.begin();
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
    //std::cout << "Number of cors: " << numberOfCores;
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
    std::pair<std::vector<std::map<int, std::vector<int> > >*, std::map<int, std::vector<int> >* > index_signatureStorage =
                     _computeInverseIndex(numberOfHashFunctions, instance_featureVector,
                                     blockSize, maxBinSize, numberOfCores, chunkSize, NULL, NULL);
    std::vector<std::map<int, std::vector<int> > >* inverseIndex = index_signatureStorage.first;
    std::map<int, std::vector<int> >* signatureStorage = index_signatureStorage.second;
    size_t adressOfInverseIndex = reinterpret_cast<size_t>(inverseIndex);
    size_t adressOfSignatureStorage= reinterpret_cast<size_t>(signatureStorage);

   // std::cout << "InverseIndexPointer_Build: "  << adressOfInverseIndex << std::endl << std::flush;
    //std::cout << "First element in first map BEVOR restore: " << inverseIndex->at(50).begin()->second.size() << std::endl << std::flush;

    PyObject * pointerToInverseIndex = Py_BuildValue("k", adressOfInverseIndex);
    PyObject * pointerToSignatureStorage = Py_BuildValue("k", adressOfSignatureStorage);
    PyObject * outList = PyList_New(2);
    PyList_SetItem(outList, 0, pointerToInverseIndex);
    PyList_SetItem(outList, 1, pointerToSignatureStorage);
    return outList;
}
// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeSignature(PyObject* self, PyObject* args)
{
    int numberOfHashFunctions, blockSize, numberOfCores, chunkSize;
    std::map< int, std::vector<int> > instanceFeatureVector;
    size_t addressSignatureStorage;

    PyObject * listInstancesObj;
    PyObject * listFeaturesObj;
    PyObject * listObjFeature;
    PyObject * intInstancesObj;
    PyObject * intFeaturesObj;
    int instanceOld = -1;
  
    if (!PyArg_ParseTuple(args, "iO!O!iiik", &numberOfHashFunctions,
                        &PyList_Type, &listInstancesObj,
                        &PyList_Type, &listFeaturesObj, &blockSize, &numberOfCores, &chunkSize,
                        &addressSignatureStorage))
        return NULL;

    int numLinesInstances = PyList_Size(listInstancesObj);
    if (numLinesInstances < 0)	return NULL;

    int numLinesFeatures = PyList_Size(listFeaturesObj);
    if (numLinesFeatures < 0)	return NULL;


    int insert = 0;
    std::vector<int> featureIds;
   // std::cout << "BEvor trans" << std::endl << std::flush;
    // parse from python list to c++ vector
    for (int i = 0; i < numLinesInstances; ++i) {
        intInstancesObj = PyList_GetItem(listInstancesObj, i);
        intFeaturesObj = PyList_GetItem(listFeaturesObj, i);
        int featureValue;
        int instanceValue;

        PyArg_Parse(intInstancesObj, "i", &instanceValue);
        PyArg_Parse(intFeaturesObj, "i", &featureValue);

        if (instanceOld == instanceValue) {
            featureIds.push_back(featureValue);
            instanceOld = instanceValue;
            if (i == numLinesInstances-1) {
                instanceFeatureVector[instanceOld] = featureIds;
            }
        } else {
            if (instanceOld != -1 ) {
                instanceFeatureVector[instanceOld] = featureIds;
            }
            featureIds.clear();
//            instanceFeatureVector.push_back(featureIds);
            instanceOld = instanceValue;
        }
    }
    // get pointer to signature storage
    std::map<int, std::vector<int> >* signatureStorage =
            reinterpret_cast<std::map<int, std::vector<int> >* >(addressSignatureStorage);
    // compute in c++
    std::vector< std::vector<int> >signatures = _computeSignature(numberOfHashFunctions, instanceFeatureVector,
                                                                                    blockSize,numberOfCores, chunkSize,
                                                                                    *signatureStorage);
    int sizeOfSignature = signatures.size();
    PyObject * outListInstancesObj = PyList_New(sizeOfSignature);
    for (int i = 0; i < sizeOfSignature; ++i) {
    int sizeOfFeature = signatures[i].size();
        PyObject * outListFeaturesObj = PyList_New(sizeOfFeature);
        for (int j = 0; j < sizeOfFeature; ++j) {
            PyObject* value = Py_BuildValue("i", signatures[i][j]);
            PyList_SetItem(outListFeaturesObj, j, value);
        }
        PyList_SetItem(outListInstancesObj, i, outListFeaturesObj);
    }

    return outListInstancesObj;
}


// parse python call to c++; execute c++ and parse it back to python
static PyObject* computeNeighborhood(PyObject* self, PyObject* args)
{ // numberOfHashFunktions, instances_list, features_list, block_size,
    // number_of_cores, chunk_size, size_of_neighborhood, MAX_VALUE, minimalBlocksInCommon,
    // excessFactor, maxBinSize, inverseIndex

    size_t addressInverseIndex;
    size_t addressSignatureStorage;

    int numberOfHashFunctions, blockSize, numberOfCores, chunkSize,
    sizeOfNeighborhood, MAX_VALUE, minimalBlocksInCommon, maxBinSize,
    maximalNumberOfHashCollisions, excessFactor;
    std::map< int, std::vector<int> > instanceFeatureVector;

    PyObject * listInstancesObj;
    PyObject * listFeaturesObj;
    PyObject * listObjFeature;
    PyObject * intInstancesObj;
    PyObject * intFeaturesObj;
    int instanceOld = -1;
  
    if (!PyArg_ParseTuple(args, "iO!O!iiiiiiiiikk", &numberOfHashFunctions,
                        &PyList_Type, &listInstancesObj,
                        &PyList_Type, &listFeaturesObj, &blockSize, 
                        &numberOfCores, &chunkSize, &sizeOfNeighborhood, 
                        &MAX_VALUE, &minimalBlocksInCommon, &maxBinSize,
                        &maximalNumberOfHashCollisions, &excessFactor, 
                        &addressInverseIndex, &addressSignatureStorage))
        return NULL;

    int numLinesInstances = PyList_Size(listInstancesObj);
    if (numLinesInstances < 0)  return NULL;

    int numLinesFeatures = PyList_Size(listFeaturesObj);
    if (numLinesFeatures < 0)   return NULL;


    int insert = 0;
    std::vector<int> featureIds;
   // std::cout << "BEvor trans" << std::endl << std::flush;
    // parse from python list to c++ vector
    for (int i = 0; i < numLinesInstances; ++i) {
        intInstancesObj = PyList_GetItem(listInstancesObj, i);
        intFeaturesObj = PyList_GetItem(listFeaturesObj, i);
        int featureValue;
        int instanceValue;

        PyArg_Parse(intInstancesObj, "i", &instanceValue);
        PyArg_Parse(intFeaturesObj, "i", &featureValue);

        if (instanceOld == instanceValue) {
            featureIds.push_back(featureValue);
            instanceOld = instanceValue;
            if (i == numLinesInstances-1) {
                instanceFeatureVector[instanceOld] = featureIds;
            }
        } else {
            if (instanceOld != -1 ) {
                instanceFeatureVector[instanceOld] = featureIds;
            }
            featureIds.clear();
//            instanceFeatureVector.push_back(featureIds);
            instanceOld = instanceValue;
        }
    }

    // get pointer to inverse index
    std::vector<std::map<int, std::vector<int> > >* inverseIndex =
            reinterpret_cast<std::vector<std::map<int, std::vector<int> > >* >(addressInverseIndex);
    // get pointer to signature storage
    std::map<int, std::vector<int> >* signatureStorage =
            reinterpret_cast<std::map<int, std::vector<int> >* >(addressSignatureStorage);
    // compute signatures of the instances
    std::vector< std::vector<int> >signatures = _computeSignature(numberOfHashFunctions, instanceFeatureVector,
                                                                blockSize,numberOfCores, chunkSize, *signatureStorage);

    // compute the k-nearest neighbors
    std::pair<std::vector< std::vector<int> > , std::vector< std::vector<float> > > neighborsDistances = 
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
            PyObject* valueNeighbor = Py_BuildValue("i", neighborsDistances.first[i][j]);
            PyList_SetItem(innerListNeighbors, j, valueNeighbor);
            // std::cout << neighborsDistances.first[i][j] << std::endl <<  std::flush;
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

    int numberOfHashFunctions, blockSize, maxBinSize, numberOfCores, chunkSize;
    std::map<int, std::vector<int> > instance_featureVector;
    PyObject * instancesListObj;
    PyObject * featuresListObj;
    PyObject * instanceIntObj;
    PyObject * featureIntObj;

    if (!PyArg_ParseTuple(args, "iO!O!iiiikk", &numberOfHashFunctions, &PyList_Type, &instancesListObj, &PyList_Type,
                            &featuresListObj, &blockSize, &maxBinSize, &numberOfCores, &chunkSize, &addressInverseIndex,
                            &addressSignatureStorage))
        return NULL;
    //std::cout << "Number of cors: " << numberOfCores;
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

    // get pointer to inverse index
    std::vector<std::map<int, std::vector<int> > >* inverseIndex =
            reinterpret_cast<std::vector<std::map<int, std::vector<int> > >* >(addressInverseIndex);
    // get pointer to inverse index
    std::map<int, std::vector<int> >* signatureStorage =
            reinterpret_cast<std::map<int, std::vector<int> >* >(addressSignatureStorage);
    // compute inverse index in c++
    std::pair<std::vector<std::map<int, std::vector<int> > >*, std::map<int, std::vector<int> >* > index_signatureStorage =
                     _computeInverseIndex(numberOfHashFunctions, instance_featureVector,
                                     blockSize, maxBinSize, numberOfCores, chunkSize, inverseIndex, signatureStorage);
    inverseIndex = index_signatureStorage.first;
    signatureStorage = index_signatureStorage.second;
    size_t adressOfInverseIndex = reinterpret_cast<size_t>(inverseIndex);
    size_t adressOfSignatureStorage= reinterpret_cast<size_t>(signatureStorage);

   // std::cout << "InverseIndexPointer_Build: "  << adressOfInverseIndex << std::endl << std::flush;
    //std::cout << "First element in first map BEVOR restore: " << inverseIndex->at(50).begin()->second.size() << std::endl << std::flush;

    PyObject * pointerToInverseIndex = Py_BuildValue("k", adressOfInverseIndex);
    PyObject * pointerToSignatureStorage = Py_BuildValue("k", adressOfSignatureStorage);
    PyObject * outList = PyList_New(2);
    PyList_SetItem(outList, 0, pointerToInverseIndex);
    PyList_SetItem(outList, 1, pointerToSignatureStorage);
    return outList;
}



// definition of avaible functions for python and which function parsing fucntion in c++ should be called.
static PyMethodDef IntHashMethods[] = {
    {"computeIntHash", computeIntHash, METH_VARARGS, "Calculate a hash for given key, modulo and seed."},
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
    (void) Py_InitModule("_hashUtility", IntHashMethods);
}
