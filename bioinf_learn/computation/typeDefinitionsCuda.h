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

struct cudaInstance{
    uint x;
    float y;
};

struct cudaInstanceVector {
    cudaInstance* instance;
};
#endif // TYPE_DEFINTIONS_CUDA_H