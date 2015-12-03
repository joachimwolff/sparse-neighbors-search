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
    size_t mM;
    size_t mK;
    size_t mQ;
    BloomierHash* mBloomierHash;
    size_t mBitVectorSize;
    // BloomierHash mBloomierHash;
    // size_t mByteSize;
    OrderAndMatchFinder* mOrderAndMatchFinder;
    // std::map<size_t, size_t> mKeyDict;
    Encoder* mEncoder;
    size_t mPiIndex;

  public:      
    BloomierFilter(size_t pM, size_t pK, size_t pQ, size_t pHashSeed);
    
    ~BloomierFilter();
    
    size_t getByteSize(size_t pQ);
    bloomierTable* getTable();
    void setTable(bloomierTable* pTable);
    vvsize_t_p* getValueTable();
    void setValueTable(vvsize_t_p* pTable);
    void xorBitVector(bitVector* pResult, bitVector* pInput);
    void xorOperation(bitVector* pValue, bitVector* pM, vsize_t* pNeighbors);
    vsize_t* get(size_t pKey);
    bool set(size_t pKey, size_t pValue);
    void create(vsize_t* pKeys, vvsize_t_p* pValues, size_t piIndex = 0);
    void check();
    // std::string tableToString();
    // std::pair<vsize_t, vsize_t> stringToTable(std::string pString);
};
#endif // BLOOMIER_FILTER_H