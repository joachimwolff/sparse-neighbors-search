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
    std::unordered_map<size_t, vsize_t* >* mStoredNeighbors;
    std::unordered_map<size_t, bitVector*>*  mStoredMasks;
    void create( size_t pKey, size_t pValue);
    
  public:      
    BloomierFilter(const size_t pM, const size_t pK, const size_t pQ, const size_t pHashSeed, const size_t pMaxBinSize);
    
    ~BloomierFilter();
    
    size_t getByteSize(const size_t pQ);
    // bloomierTable* getTable();
    // void setTable(bloomierTable* pTable);
    // vvsize_t* getValueTable();
    // void setValueTable(vvsize_t* pTable);
    void xorBitVector(bitVector* pResult, const bitVector* pInput);
    void xorOperation(bitVector* pValue, const bitVector* pM, const vsize_t* pNeighbors);
    vsize_t* get(const size_t pKey);
    bool set(const size_t pKey, const size_t pValue);
};
#endif // BLOOMIER_FILTER_H