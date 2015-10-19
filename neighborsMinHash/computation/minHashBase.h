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
#include <vector>
#include <map> 
#include <unordered_map>

#include <Python.h>
#include <boost/numeric/mtl/mtl.hpp>
#include "inverseIndex.h"


typedef std::vector<size_t> vsize_t;
typedef std::unordered_map< size_t, vsize_t > umapVector;
typedef std::vector<vsize_t > vvsize_t;
typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<std::map<size_t, size_t> > vmSize_tSize_t;

typedef compressed2D<float, mat::parameters<tag::col_major> > csrMatrix;

struct csr_neighborhood {
  vvsize_t instances;
  vvfloat features;
  vvfloat data;
};
struct raw_data {
  csrMatrix* matrixData;
  umapVector* inverseIndexData;
};

struct uniqueElement {
  vsize_t instances;
  vsize_t features;
};

struct neighborhood {
  vvsize_t* neighbors;
  vvfloat* distances;
};

typedef std::unordered_map<size_t, uniqueElement > umap_uniqueElement;
#define MAX_VALUE 2147483647

raw_data parseRawData(PyObject * instancesListObj, PyObject * featuresListObj, PyObject * dataListObj,
                                              size_t maxNumberOfInstances, size_t maxNumberOfFeatures) {
    PyObject * instanceSize_tObj;
    PyObject * featureSize_tObj;
    PyObject * dataSize_tObj;


    umapVector* inverseIndexData = new umapVector();
    // create sparse matrix and inserter
    csrMatrix originalData(maxNumberOfInstances, maxNumberOfFeatures);
    mat::inserter< csrMatrix >* insertElements = new mat::inserter< csrMatrix >(originalData);

    vsize_t featureIds;
    size_t instanceOld = 0;
    size_t sizeOfFeatureVector = PyList_Size(instancesListObj);

    for (size_t i = 0; i < sizeOfFeatureVector; ++i) {
        instanceSize_tObj = PyList_GetItem(instancesListObj, i);
        featureSize_tObj = PyList_GetItem(featuresListObj, i);
        dataSize_tObj = PyList_GetItem(dataListObj, i);
        size_t featureValue;
        size_t instanceValue;

        PyArg_Parse(instanceSize_tObj, "k", &instanceValue);
        PyArg_Parse(featureSize_tObj, "k", &featureValue);
        PyArg_Parse(dataSize_tObj, "k", &featureValue);

        if (instanceOld == instanceValue) {
            featureIds.push_back(featureValue);
            (*insertElements)[instanceValue][featureValue] << dataSize_tObj;
            instanceOld = instanceValue;
            if (i == sizeOfFeatureVector-1) {
                (*inverseIndexData)[instanceValue] = featureIds;
            }
        } else {
            if (instanceOld != MAX_VALUE) {
                (inverseIndexData*)[instanceOld] = featureIds;
            }
            featureIds.clear();
            featureIds.push_back(featureValue);
            insertElements[instanceValue][featureValue] << dataSize_tObj;

            instanceOld = instanceValue;
        }
    }
    // delete inserter to get sparse matrix accessible
    delete insertElements;

    raw_data returnValues;
    returnValues.matrixData = &originalData;
    returnValues.inverseIndexData = &inverseIndexData;

    return returnValues;
}

class MinHashBase {
  protected:
    InverseIndex* mInverseIndex;
    csrMatrix* mOriginalData;

	  csr_neighborhood computeNeighborhood();
    csr_neighborhood computeExactNeighborhood();
  	csr_neighborhood computeNeighborhoodGraph();

  	MinHashBase(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions);

  	~MinHashBase();
    // Calculate the inverse index for the given instances.
    void fit(umapVector* instanceFeatureVector); 
    // Extend the inverse index with the given instances.
    void partialFit(); 
    // Calculate k-nearest neighbors.
    neighborhood kneighbors(rawData pRawData, size_t pNneighbors, size_t pReturnDistance=1, size_t pFast=0); 
    // Calculate k-nearest neighbors as a graph.
    neighborhood kneighborsGraph();
    // Fits and calculates k-nearest neighbors.
    neighborhood fitKneighbors();
    // Fits and calculates k-nearest neighbors as a graph.
    neighborhood fitKneighborsGraph();
    // Cut the neighborhood to the length of k-neighbors
    void cutKneighborhood(neighborhood* pNeighborhood, size_t pKneighborhood);

  public:

    void set_mOriginalData(csrMatrix* pOriginalData) {
      mOriginalData = pOriginalData;
      return;
    }
};