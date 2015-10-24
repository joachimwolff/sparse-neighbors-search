#ifndef TYPE_DEFINTIONS_H
#define TYPE_DEFINTIONS_H

#include <vector>
#include <map> 
#include <unordered_map>

// #include <boost/numeric/mtl/mtl.hpp>

#define MAX_VALUE 2147483647

typedef std::vector<size_t> vsize_t;
typedef std::vector<int> vint;
typedef std::unordered_map< size_t, vsize_t > umapVector;
typedef std::vector<vsize_t > vvsize_t;
typedef std::vector<vint > vvint;
typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<std::map<size_t, size_t> > vmSize_tSize_t;

// typedef mtl::compressed2D<float, mtl::mat::parameters<mtl::tag::col_major> > csrMatrix;

struct csrNeighborhood {
  vvsize_t instances;
  vvfloat features;
  vvfloat data;
};
struct rawData {
  // csrMatrix* matrixData;
  umapVector* inverseIndexData;
};

struct uniqueElement {
  vsize_t instances;
  vsize_t features;
};

struct neighborhood {
  vvint* neighbors;
  vvfloat* distances;
};

typedef std::unordered_map<size_t, uniqueElement > umap_uniqueElement;

#endif // TYPE_DEFINTIONS_H