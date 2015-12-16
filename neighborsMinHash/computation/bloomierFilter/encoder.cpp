#include "encoder.h"

Encoder::Encoder(size_t pBitVectorSize) {
	mBitVectorSize = pBitVectorSize;
	mMaskingValues = new vsize_t(pBitVectorSize);
	size_t mask = 0b11111111;
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		(*mMaskingValues)[i] = mask << (8*i);
	}
}
bitVector* Encoder::encode(size_t pValue) {
	bitVector* value = new bitVector(mBitVectorSize);
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		value[i] = (pValue & (*mMaskingValues)[i]) >> (8*i);
	}
	return value;
}
size_t Encoder::decode(bitVector* pValue, size_t pSizeOfValue) {
	size_t value = 0;
	size_t castValue = 0;
	for (size_t i = 0; i < pSizeOfValue; ++i) {
		castValue = pValue[i];
		value = value | (castValue << (8*i));
	}
	return value;
}
vsize_t* Encoder::getMaskingValues() {
	return mMaskingValues;
}