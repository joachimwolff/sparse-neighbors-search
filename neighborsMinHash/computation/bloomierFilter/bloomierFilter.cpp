#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(size_t pM, size_t pK, size_t pQ, size_t pHashSeed){
	mM = pM;;
	mK = pK;
	mQ = pQ;
    mHashSeed = pHashSeed;
	mBitVectorSize = ceil(pQ / static_cast<float>(CHAR_BIT));
	mBloomierHash = new BloomierHash(pM, pK, mBitVectorSize, pHashSeed);
    
	mOrderAndMatchFinder = new OrderAndMatchFinder(pM, pK, mBloomierHash);
	mTable = new bloomierTable(pM);
	mValueTable = new vvsize_t(pM);
	for (size_t i = 0; i < pM; ++i) {
		(*mTable)[i] = new bitVector(mBitVectorSize, 0);
		// (*mValueTable)[i] = new vsize_t();
	}
	
    mEncoder = new Encoder(mBitVectorSize);
	mPiIndex = 0;
	// mOrderAndMatchFinder->find(pKeys);
	
}

BloomierFilter::~BloomierFilter(){

}
void BloomierFilter::check() {
	// std::cout << __LINE__ << std::endl;
	// size_t sumTable = 0;
	// for(size_t i = 0; i < mTable->size(); ++i) {
    //     try {
	// 	  sumTable += (*mTable)[i]->size();
    //     } catch(int e) {
    //         std::cout << "i: " << i << std::endl;
    //     }
	// }
	// std::cout << __LINE__ << std::endl;
	// size_t sumValueTable = 0;
	// for(size_t i = 0; i < mValueTable->size(); ++i) {
	// 	sumValueTable += (*mValueTable)[i]->size();
	// }
	// std::cout << "sumTable: "  << sumTable << std::endl;
	// std::cout << "sumValueTable: "  << sumValueTable << std::endl;
	
	// std::cout << __LINE__ << std::endl;
	
}
bloomierTable* BloomierFilter::getTable() {
	return mTable;
}

void BloomierFilter::setTable(bloomierTable* pTable) {
    if (mTable != NULL) {
        for (size_t i = 0; i < mTable->size(); ++i) {
            delete (*mTable)[i];
        }
        delete mTable;
    }
	mTable = pTable;
}

vvsize_t* BloomierFilter::getValueTable() {
	return mValueTable;
}
void BloomierFilter::setValueTable(vvsize_t* pTable) {
	mValueTable = pTable;
} 

void BloomierFilter::xorOperation(bitVector* pValue, bitVector* pMask, vsize_t* pNeighbors) {
    this->xorBitVector(pValue, pMask);
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		if (*it < mTable->size()) {
			this->xorBitVector(pValue, (*mTable)[(*it)]);
		}
	}
}
vsize_t* BloomierFilter::get(size_t pKey) {
    
    size_t valueSeed = mOrderAndMatchFinder->getSeed(pKey);
    if (valueSeed == MAX_VALUE) {
        return new vsize_t();
    } else if (valueSeed == MAX_VALUE - 1) {
        valueSeed = mHashSeed;
    }
    
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey, valueSeed);
	bitVector* mask = mBloomierHash->getMask(pKey);
		
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder->decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
			
            delete neighbors;
			delete mask;
			delete valueToGet;
			return &(*mValueTable)[L];
		}
	}
	
    delete neighbors;
	delete mask;
	delete valueToGet;
	return new vsize_t();
}
bool BloomierFilter::set(size_t pKey, size_t pValue) {
    // size_t seed = 0;
    size_t valueSeed = mOrderAndMatchFinder->getSeed(pKey);
    if (valueSeed == MAX_VALUE) {
        std::unordered_map<size_t, vsize_t >* keyValue = new std::unordered_map<size_t, vsize_t >();
        vsize_t value;
        value.push_back(pValue);
        (*keyValue)[pKey] = value;
		this->create(keyValue, mPiIndex);
		return true;
    } else if (valueSeed == MAX_VALUE - 1) {
        valueSeed = mHashSeed;
    }
    
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey, valueSeed);
	bitVector* mask = mBloomierHash->getMask(pKey);
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);
	size_t h = mEncoder->decode(valueToGet);
	
	delete mask;
	delete valueToGet;
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		delete neighbors;
		if (L < mValueTable->size()) {
			vsize_t* v = &((*mValueTable)[L]);
			v->push_back(pValue);
			return true;
		}
	}
	return false;
}

void BloomierFilter::create( std::unordered_map<size_t, vsize_t >* pKeyValue, size_t piIndex) {
	vsize_t* keys = new vsize_t(pKeyValue->size());
	size_t i = 0;
    for (auto it = pKeyValue->begin(); it != pKeyValue->end(); ++it) {
        (*keys)[i] = it->first;
		++i;
    }
	// mOrderAndMatchFinder = new OrderAndMatchFinder(pM, pK, mBloomierHash);
	
    mOrderAndMatchFinder->find(keys);
    vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();
    // std::cout << "size if inout vector: " << pKeyValue->size() << std::endl;
		// std::cout << "size of piVector: " << piVector->size() << std::endl; 
	
	for (size_t i = mPiIndex; i < piVector->size(); ++i) {
		// std::cout << "pi element: " << (*piVector)[i] << std::endl;
		// std::cout << "tau element: " << (*tauVector)[i] << std::endl;

        size_t key = (*piVector)[i];
		vsize_t* neighbors = mBloomierHash->getKNeighbors(key, mOrderAndMatchFinder->getSeed(key));
		bitVector* mask = mBloomierHash->getMask(key);
        // std::cout << "\tmask: ";
		// for (size_t i = 0; i < mask->size(); ++i) {
			// std::cout << static_cast<size_t>((*mask)[i]) << " foo";
		// }
		// std::cout << std::endl;
        
		size_t l = (*tauVector)[i];
		size_t L = (*neighbors)[l];
		// std::cout << "neighbor: " << (*neighbors)[l] << std::endl << std::endl;
		bitVector* encodeValue = mEncoder->encode(l);
        // this->xorOperation(encodeValue, mask, neighbors);
		this->xorBitVector((*mTable)[L], encodeValue);
		this->xorBitVector((*mTable)[L], mask);
		for (size_t j = 0; j < neighbors->size(); ++j) {
			if (j != l) {
				this->xorBitVector((*mTable)[L], (*mTable)[(*neighbors)[j]]);
			}
		}
        // for (size_t j = 0; j < (*pValues)[i-piIndex]->size(); ++j) {
        (*mValueTable)[L] = (*pKeyValue)[key];
            // (*mValueTable)[L]->push_back((*pValues)[i-piIndex]->operator[](j));
        // }
		delete neighbors;
		delete mask;
		delete encodeValue;
	}
	mPiIndex += pKeyValue->size();
	// delete pKeys;
	// for (size_t i = 0; i < pValues->size(); ++i) {
	// 	delete (*pValues)[i];
	// }
	// delete pValues;
	// delete piVector;
	// delete tauVector;
	// delete pKeyValue;
	
}

void BloomierFilter::xorBitVector(bitVector* pResult, bitVector* pInput) {
	size_t length = std::min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}