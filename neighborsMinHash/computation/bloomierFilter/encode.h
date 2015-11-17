#include "../typeDefinitions.h"
#ifndef ENCODE_H
#define ENCODE_H
class Encode {
  public:
  	vsize_t* encode(size_t value, size_t width=1);
  	size_t decode(size_t value);
  	vsize_t* size_tToByteArray(size_t pValue);
  	size_t byteArrayToSize_t(vsize_t* pArray);
};
#endif // ENCODE_H