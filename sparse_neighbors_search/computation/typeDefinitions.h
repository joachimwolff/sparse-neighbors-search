/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/
 
#ifndef TYPE_DEFINTIONS_H
#define TYPE_DEFINTIONS_H

#include "typeDefinitionsBasic.h"
#include "sparseMatrix.h"

struct rawData {
  SparseMatrixFloat* matrixData;
  umapVector* inverseIndexData;
};

#endif // TYPE_DEFINTIONS_H