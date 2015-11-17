#include "encode.h"

vsize_t* Encode::encode(size_t value, size_t width=1) {
	vsize_t result = this->size_tToByteArray(value);
	vsize_t* val;
	if width < 4:
		val.= new vsize_t(4-width);
		for (size_t i = 0; i < val.size(); ++i) {
			(*val)[i] = result[i+4-width];
		}
	else {
		val = new vsize_t(width - 4 + result.size(), 0);
		for (size_t i = width - 4; i < val.size(); ++i) {
			val[i] = result[i-width - 4];
		}
	}
	return val;
}
size_t Encode::decode(size_t value) {
	return this->byteArrayToSize_t(value);
}

vsize_t* Encode::size_tToByteArray(size_t pValue) {
	vsize_t* byteArray = new vsize_t(4);
	(*byteArray)[0] = value >> 24 & 0x000000FF;
	(*byteArray)[1] = value >> 16 & 0x000000FF;
	(*byteArray)[2] = value >> 8 & 0x000000FF;
	(*byteArray)[3] = value >> 0 & 0x000000FF;
	return byteArray;
}

size_t Encode::byteArrayToSize_t(vsize_t* pArray) {
	vsize_t* val;
	if (pArray->size() < 4) {
		val = new vsize_t(4, 0);
		for (size_t i = 4 - pArray->size(); i < val.size(); ++i) {
			val[i] = (*pArray)[i-pArray->size()];
		}
	} else {
		val = pArray;
	}
	return (*val)[0] << 24 + (*val)[1] << 16 + (*val)[2] << 8 + (*val)[3];
}