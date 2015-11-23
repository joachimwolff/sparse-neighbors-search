#include "../typeDefinitions.h"
#ifndef ENCODER_H
#define ENCODER_H
class Encoder {
  public:
  	bitVector* encode(size_t value);
  	size_t decode(bitVector value);
  	
};
#endif // ENCODER_H