#include "../typeDefinitions.h"
#include "bloomierHash.h"
// #include "bloomierHash.h"
// #include "orderAndMatchFinder.h"
// #include "encoder.h"
#ifndef BLOOMIER_FILTER_H
#define BLOOMIER_FILTER_H
class BloomierFilter {
  private:
    bloomierTable* mTable;
    vsize_t* mValueTable;
    size_t mM;
    size_t mK;
    size_t mQ;
    BloomierHash mBloomierHash;
    // BloomierHash mBloomierHash;
    // size_t mByteSize;
    OrderAndMatchFinder mOrderAndMatchFinder;
    // std::map<size_t, size_t> mKeyDict;
    Encoder mEncoder;

  public:      
    BloomierFilter(size_t pM, size_t pK, size_t pQ);
    
    ~BloomierFilter();
    
    size_t getByteSize(size_t pQ);
    vsize_t* getTable();
    void setTable(vsize_t pTable);
    size_t getValueTable();
    void setValueTable(vsize_t pTable);
    void byteArrayXor(vsize_t* pResult, vsize_t* pInput);
    size_t xorOperation(vsize_t* pValue, vsize_t* pM, vsize_t* pNeighbors);
    vsize_t get(size_t pKey);
    bool set(size_t pKey, size_t pValue);
    void create(std::map<size_t, size_t> pAssignment);
    std::string tableToString();
    std::pair<vsize_t, vsize_t> stringToTable(std::string pString);
};
#endif // BLOOMIER_FILTER_H