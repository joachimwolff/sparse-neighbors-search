/**
 Copyright 2016, 2017, 2018, 2019, 2020 Joachim Wolff
 PhD Thesis

 Copyright 2015, 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/
#ifndef TYPE_DEFINTIONS_BASIC_H
#define TYPE_DEFINTIONS_BASIC_H

#include <vector>
#include <map> 
#include <unordered_map>
#include <utility>
#include <limits>

#define MAX_VALUE std::numeric_limits<uint32_t>::max()


typedef std::vector< size_t > vsize_t;
typedef std::vector< int > vint;
typedef std::vector< float > vfloat;

typedef std::vector< vsize_t > vvsize_t;
typedef std::vector< vsize_t* > vvsize_t_p;

typedef std::vector< vint > vvint;
typedef std::vector< vfloat > vvfloat;

typedef std::unordered_map< size_t, vsize_t > umapVector;
typedef std::unordered_map< size_t, vsize_t* > umapVector_ptr;

typedef std::vector< std::map< size_t, size_t > > vmSize_tSize_t;
typedef std::vector< umapVector > vector__umapVector;
typedef std::vector< umapVector_ptr* > vector__umapVector_ptr;


struct uniqueElement {
  vsize_t* instances;
  vsize_t* signature;
};

struct neighborhood {
  vvsize_t* neighbors;
  vvfloat* distances;
};

typedef std::unordered_map< size_t, uniqueElement > umap_uniqueElement;


struct sortMapFloat {
    size_t key;
    float val;
};


typedef unsigned char bitVector;

typedef std::vector<bitVector*> bloomierTable;

struct distributionInverseIndex {
    vsize_t numberOfCreatedHashValuesPerHashFunction;
    vsize_t meanForNumberOfValuesPerHashValue;
    vsize_t standardDeviationForNumberOfValuesPerHashValue;
    std::map<size_t, size_t> totalCountForOccurenceOfHashValues;
    
    size_t mean;
    size_t standardDeviation;
};

static bool mapSortDescByValueFloat(const sortMapFloat& a, const sortMapFloat& b) {
        return a.val > b.val;
};

static bool mapSortAscByValueFloat(const sortMapFloat& a, const sortMapFloat& b) {
        return a.val < b.val;
};

#endif // TYPE_DEFINTIONS_BASIC_H
