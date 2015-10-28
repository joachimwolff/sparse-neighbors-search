#ifndef TYPE_DEFINTIONS_H
#define TYPE_DEFINTIONS_H

#include <vector>
#include <map> 
#include <unordered_map>

// #include <eigen3/Eigen/Sparse>
// #include <boost/numeric/mtl/mtl.hpp>
// #include <armadillo>
#include "sparseMatrix.h"
#define MAX_VALUE 2147483647

typedef std::vector<size_t> vsize_t;
typedef std::vector<int> vint;
typedef std::unordered_map< size_t, vsize_t > umapVector;
typedef std::vector<vsize_t > vvsize_t;
typedef std::vector<vint > vvint;
typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<std::map<size_t, size_t> > vmSize_tSize_t;

// typedef Eigen::Triplet<double> triplet;
// typedef arma::sp_fmat SparseMatrixFloat;
// typedef arma::umat umat;
// typedef arma::vec vec;
// typedef arma::endr endr;
// typedef Eigen::SparseVector<float> SparseVectorFloat;


struct csrNeighborhood {
  vvsize_t instances;
  vvfloat features;
  vvfloat data;
};
struct rawData {
  SparseMatrixFloat* matrixData;
  umapVector* inverseIndexData;
};

struct uniqueElement {
  vsize_t instances;
  vsize_t signature;
};

struct neighborhood {
  vvint* neighbors;
  vvfloat* distances;
};

typedef std::unordered_map<size_t, uniqueElement > umap_uniqueElement;


#endif // TYPE_DEFINTIONS_H