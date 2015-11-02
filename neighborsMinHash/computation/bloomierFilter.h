#include "typeDefinitions.h"
#ifndef BLOOMIER_FILTER_H
#define BLOOMIER_FILTER_H
class BloomierFilter {
  private:
    vint mTable1;
    vint mTable2;
    size_t mHashSeed;
    size_t mM;
    size_t mK;
    size_t mQ;
  public:      
    BloomierFilter(size_t pHashSeed, size_t pM, size_t pK, size_t pQ) {

    };
    ~BloomierFilter() {

    };
    vint* lookup(size_t pKey);
    void setValue(size_t pKey, size_t pValue);
    vint* findMatch(size_t pHashSeed, vint* pSubset);
    void create(std::map<size_t, size_t> pAssignment);

};
#endif // BLOOMIER_FILTER_H