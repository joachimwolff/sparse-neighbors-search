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
#ifndef TYPE_DEFINTIONS_BASIC_H
#define TYPE_DEFINTIONS_BASIC_H

#include <vector>
#include <map> 
#include <unordered_map>
#include <utility>
#include <limits>

#define MAX_VALUE std::numeric_limits<int>::max()

typedef std::vector< size_t > vsize_t;
typedef std::vector< int > vint;
typedef std::vector< float > vfloat;

typedef std::vector< vsize_t > vvsize_t;
typedef std::vector< vsize_t* > vvsize_t_p;

typedef std::vector< vint > vvint;
typedef std::vector< vfloat > vvfloat;

typedef std::unordered_map< size_t, vsize_t > umapVector;

typedef std::vector< std::map< size_t, size_t > > vmSize_tSize_t;
typedef std::vector< umapVector > vector__umapVector;

struct uniqueElement {
  vsize_t* instances;
  vsize_t* signature;
};

struct neighborhood {
  vvint* neighbors;
  vvfloat* distances;
};

typedef std::unordered_map< size_t, uniqueElement* > umap_uniqueElement;


struct sortMapFloat {
    size_t key;
    float val;
};

typedef unsigned char bitVector;

typedef std::vector<bitVector*> bloomierTable;

static bool mapSortDescByValueFloat(const sortMapFloat& a, const sortMapFloat& b) {
        return a.val > b.val;
};
static bool mapSortAscByValueFloat(const sortMapFloat& a, const sortMapFloat& b) {
        return a.val < b.val;
};

#endif // TYPE_DEFINTIONS_BASIC_H