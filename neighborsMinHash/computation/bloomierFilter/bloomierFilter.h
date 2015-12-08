#include <climits>
#include "../typeDefinitions.h"
#include "bloomierHash.h"
// #include "bloomierHash.h"
#include "orderAndMatchFinder.h"
#include "encoder.h"
#ifndef BLOOMIER_FILTER_H
#define BLOOMIER_FILTER_H
class BloomierFilter {
  private:
    bloomierTable* mTable;
    vvsize_t_p* mValueTable;
    size_t mModulo;
	size_t mNumberOfElements;
    // size_t mQ;
    size_t mHashSeed;
    BloomierHash* mBloomierHash;
    size_t mBitVectorSize;
    OrderAndMatchFinder* mOrderAndMatchFinder;
    Encoder* mEncoder; 
    size_t mPiIndex;
    size_t mMaxBinSize;

    void create( size_t pKey, size_t pValue);
    
  public:      
    BloomierFilter(size_t pM, size_t pK, size_t pQ, size_t pHashSeed, size_t pMaxBinSize);
    
    ~BloomierFilter();
    
    size_t getByteSize(size_t pQ);
    // bloomierTable* getTable();
    // void setTable(bloomierTable* pTable);
    // vvsize_t* getValueTable();
    // void setValueTable(vvsize_t* pTable);
    void xorBitVector(bitVector* pResult, bitVector* pInput);
    void xorOperation(bitVector* pValue, bitVector* pM, vsize_t* pNeighbors);
    vsize_t* get(size_t pKey);
    bool set(size_t pKey, size_t pValue);
};
#endif // BLOOMIER_FILTER_H