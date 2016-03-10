/**
 Copyright 2016 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/
#ifndef TYPE_DEFINTIONS_CUDA_H
#define TYPE_DEFINTIONS_CUDA_H
struct hits {
    size_t* instances;
    size_t size;
};

struct histogram {
    size_t* instances;
};
struct sortedHistogram {
    int2* instances;
    size_t size;
};
struct radixSortingMemory {
    int2* bucketNull;
    int2* bucketOne;
};

#endif // TYPE_DEFINTIONS_CUDA_H