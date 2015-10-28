#ifndef TYPE_DEFINTIONS_H
#define TYPE_DEFINTIONS_H

#include "typeDefinitionsBasic.h"
#include "sparseMatrix.h"

struct rawData {
  SparseMatrixFloat* matrixData;
  umapVector* inverseIndexData;
};

#endif // TYPE_DEFINTIONS_H