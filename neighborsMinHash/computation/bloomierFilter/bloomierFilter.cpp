#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(size_t pM, size_t pK, size_t pQ){
	// mHashSeed = pHashSeed;
	mM = pM;;
	mK = pK;
	mQ = pQ;
	// mKeyDict = pKeyDict;
	mBitVectorSize = ceil(pQ / static_cast<float>(CHAR_BIT));
	mBloomierHash = new BloomierHash(pM, pK, mBitVectorSize);
	mOrderAndMatchFinder = new OrderAndMatchFinder(pM, pK, pQ, mBloomierHash);
	// vsize_t orderAndMatch = mOrderAndMatchFinder.find(); // datatype?
	// mByteSize = this->getByteSize(pQ);
	mTable = new bloomierTable(pM);
	mValueTable = new vvsize_t_p(pM);
	for (size_t i = 0; i < pM; ++i) {
		(*mTable)[i] = new bitVector(mBitVectorSize, 0);
		(*mValueTable)[i] = new vsize_t();
	}
    // mValueTable = new vvsize_t_p(pM);
	
    mEncoder = new Encoder(mBitVectorSize);
	mPiIndex = 0;
    // this->create(pKeyDict, orderAndMatch);
}

BloomierFilter::~BloomierFilter(){

}

bloomierTable* BloomierFilter::getTable() {
	return mTable;
}

void BloomierFilter::setTable(bloomierTable* pTable) {
	mTable = pTable;
}

vvsize_t_p* BloomierFilter::getValueTable() {
	return mValueTable;
}
void BloomierFilter::setValueTable(vvsize_t_p* pTable) {
	mValueTable = pTable;
} 

void BloomierFilter::xorOperation(bitVector* pValue, bitVector* pMask, vsize_t* pNeighbors) {
	// std::cout << "41" << std::endl;
	
	this->xorBitVector(pValue, pMask);
	// std::cout << "44" << std::endl;

	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
	// std::cout << "47" << std::endl;

		this->xorBitVector(pValue, (*mTable)[(*it)]);
	// std::cout << "50" << std::endl;

	}
	// return pValue;
}
vsize_t* BloomierFilter::get(size_t pKey) {
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey, mK, mM);
	bitVector* mask = mBloomierHash->getMask(pKey);

	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder->decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
			return (*mValueTable)[L];
		}
	}
	return new vsize_t();
}
bool BloomierFilter::set(size_t pKey, size_t pValue) {
	std::cout << "64" << std::endl;
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey, mK, mM);
	// std::cout << "66" << std::endl;
	
	bitVector* mask = mBloomierHash->getMask(pKey);
	// std::cout << "69" << std::endl;
	

	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	// std::cout << "73" << std::endl;
	
	this->xorOperation(valueToGet, mask, neighbors);
	// std::cout << "76" << std::endl;
	

	size_t h = mEncoder->decode(valueToGet);
	std::cout << "80" << std::endl;
	
	if (h < neighbors->size()) {
	// std::cout << "83" << std::endl;
	// std::cout << "h: " << h << std::endl;	
		size_t L = (*neighbors)[h];
	// std::cout << "100" << std::endl;
	// std::cout << "L: " << L << std::endl;	

		if (L < mValueTable->size()) {
	// std::cout << "103" << std::endl;
			
			vsize_t* v = ((*mValueTable)[L]);
	// std::cout << "107" << std::endl;
			if (v == NULL) {
	// std::cout << "109" << std::endl;
				
				v = new vsize_t();
			}
	std::cout << "113" << std::endl;

			v->push_back(pValue);
	std::cout << "116" << std::endl;
			
			(*mValueTable)[L] = v;
	std::cout << "117" << std::endl;
			
			return true;
		}
	std::cout << "90" << std::endl;
		
	} else {
	std::cout << "93" << std::endl;
		
		vsize_t* keys = new vsize_t (1, pKey);
		vvsize_t_p* values = new vvsize_t_p (1);
		(*values)[0] = new vsize_t(1, pValue);
		this->create(keys, values, mPiIndex);
	std::cout << "131" << std::endl;
		
		return true;
	}
	return false;
}

void BloomierFilter::create(vsize_t* pKeys, vvsize_t_p* pValues, size_t piIndex) {
	std::cout << "139" << std::endl;
	
	mOrderAndMatchFinder->find(pKeys);
	// std::cout << "120" << std::endl;
	
    vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	// std::cout << "123" << std::endl;
	
	vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();
	// std::cout << "126" << std::endl;

	for (size_t i = piIndex; i < piVector->size(); ++i) {
		// size_t key = (*piList)[i];
		// size_t value = pAssignment[key];
		vsize_t* neighbors = mBloomierHash->getKNeighbors((*pKeys)[i], mK, mM);
		bitVector* mask = mBloomierHash->getMask((*pKeys)[i]);
		size_t l = (*tauVector)[i];
		size_t L = (*neighbors)[l];

		bitVector* encodeValue = mEncoder->encode(l);
		bitVector* valueToStore = new bitVector(mBitVectorSize, 0);
		this->xorBitVector(valueToStore, encodeValue);
		this->xorBitVector(valueToStore, mask);
		for (size_t j = 0; j < neighbors->size(); ++j) {
			if (j != l) {
				this->xorBitVector(valueToStore, (*mTable)[(*neighbors)[i]]);
			}
		}
		(*mTable)[L] = valueToStore;
		(*mValueTable)[L] = (*pValues)[i];
	}
	mPiIndex = mPiIndex + pKeys->size();
	std::cout << "171" << std::endl;
	
}

void BloomierFilter::xorBitVector(bitVector* pResult, bitVector* pInput) {
	size_t length = std::min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}