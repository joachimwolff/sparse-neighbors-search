#include "../typeDefinitions.h"
#ifndef ENCODER_H
#define ENCODER_H
class Encoder {
  private:
	size_t mBitVectorSize;
  vsize_t* mMaskingValues;
  public:
  	Encoder(size_t pBitVectorSize);
  	bitVector* encode(size_t pValue);
  	size_t decode(bitVector* pValue, size_t pSizeOfValue);
    vsize_t* getMaskingValues();
  	
};
#endif // ENCODER_H