#include "../typeDefinitions.h"
#ifndef ENCODER_H
#define ENCODER_H
class Encoder {
  private:
	size_t mBitVectorSize;
  vsize_t* mMaskingValues;
  public:
  	Encoder(size_t pBitVectorSize);
  	bitVector* encode(size_t value);
  	size_t decode(bitVector* value);
    vsize_t* getMaskingValues();
  	
};
#endif // ENCODER_H