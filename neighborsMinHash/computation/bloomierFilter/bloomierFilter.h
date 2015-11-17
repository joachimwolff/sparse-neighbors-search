#include "../typeDefinitions.h"
#include "bloomierHash.h"
#ifndef BLOOMIER_FILTER_H
#define BLOOMIER_FILTER_H
class BloomierFilter {
  private:
    vsize_t mTable1;
    vsize_t mTable2;
    size_t mHashSeed;
    size_t mM;
    size_t mK;
    size_t mQ;
    BloomierHash mBloomierHash;
    size_t mByteSize;
  public:      
    BloomierFilter(size_t pHashSeed, size_t pM, size_t pK, size_t pQ) {

    };
    ~BloomierFilter() {

    };
    vsize_t* getTable();
    void setTable(vsize_t pTable);
    size_t getValueTable();
    void setValueTable(vsize_t pTable);

    size_t xorOperation(size_t pValue, size_t pM, vsize_t pNeighbors);
    vsize_t get(size_t pKey);
    void set(size_t pKey, size_t pValue);
    void create(std::map<size_t, size_t> pAssignment, OrderAndMatchFinder pPiTau);
    std::string tableToString();
    std::pair<vsize_t, vsize_t> stringToTable(std::string pString);
};
#endif // BLOOMIER_FILTER_H