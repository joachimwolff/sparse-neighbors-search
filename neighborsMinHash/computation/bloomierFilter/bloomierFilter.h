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
    size_t mHashSeed;
    BloomierHash* mBloomierHash;
    size_t mBitVectorSize;
    OrderAndMatchFinder* mOrderAndMatchFinder;
    Encoder* mEncoder; 
    size_t mPiIndex;
    size_t mMaxBinSize;
    // std::unordered_map<size_t, vsize_t* >* mStoredNeighbors;
    void create( size_t pKey, size_t pValue);
    
  public:      
    BloomierFilter(const size_t pM, const size_t pK, const size_t pQ, const size_t pHashSeed, const size_t pMaxBinSize);
    
    ~BloomierFilter();
    
    size_t getByteSize(const size_t pQ);
    void xorBitVector(bitVector* pResult, const bitVector* pInput);
    void xorOperation(bitVector* pValue, const bitVector* pM, const vsize_t* pNeighbors);
    const vsize_t* get(const size_t pKey);
    void set(const size_t pKey, const size_t pValue);
    // bool singelton(const size_t pIndex) const;
};
#endif // BLOOMIER_FILTER_H