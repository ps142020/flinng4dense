#include "LSH.h"
#define DEBUGTIME
using namespace std;
typedef chrono::high_resolution_clock Clock;
namespace flinng {

LSH::LSH(int numHashPerFamily, int numHashFamilies, int dimension, int samsize, bool init) {

	_rangePow = numHashPerFamily,
	_numTables = numHashFamilies;
	_dimension = dimension;
	_samSize = samsize;

	printf("<<< LSH Parameters >>>\n");
	std::cout << "_rangePow " << _rangePow << std::endl;
	std::cout << "_numTables " << _numTables << std::endl;
	std::cout << "_dimension " << _dimension << std::endl;
	std::cout << "_samSize " << _samSize << std::endl;

	/* Signed random projection. */
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "Generating random number of srp hashes of dense data ...  \n";
	// Random number generation for hashing purpose - declarations.
	// Random number generation for fast random projection.
	// Reference: Anshumali Shrivastava, Rice CS
	// randBits - random bits deciding to add or subtract, contain randbits for numTable * _rangePow * samSize.
	_randBits = new short[_numTables * _rangePow * _samSize];
	// indices - selected indices to perform subtraction. Dimension same as randbits.
	_indices = new int[_numTables * _rangePow * _samSize];

	if (init) {

		int *a = new int[_dimension];
		for (int i = 0; i < _dimension; i++) {
			a[i] = i;
		}
		for (int tb = 0; tb < _numTables; tb++) {
			srand(time(0));
			for (int i = 0; i < _rangePow; i++) {
				std::random_shuffle(&a[0], &a[_dimension]);
				for (int j = 0; j < _samSize; j++) {
					_indices[tb * _rangePow * _samSize + i * _samSize + j] = a[j];
					// For 1/2 chance, assign random bit 1, or -1 to randBits.
					if (rand() % 2 == 0)
						_randBits[tb * _rangePow * _samSize + i * _samSize + j] = 1;
					else
						_randBits[tb * _rangePow * _samSize + i * _samSize + j] = -1;
				}
			}
		}
		delete[] a;
	}

}

LSH::~LSH() {

	delete[] _randBits;
	delete[] _indices;
}

void LSH::getLSHparams(short **bits, int **indices) {
	*bits = _randBits;
	*indices = _indices;
}

void LSH::getHash(unsigned int *hashIndices, float *dataVal, int numInputEntries, size_t maxNumEntries) {
#if defined DEBUGTIME
	auto begin = Clock::now();
	std::cout << "[LSH::getHash]" << std::endl;
#endif

	// Sparse srp hashes on dense data
	srp_openmp_dense_data(hashIndices, dataVal, numInputEntries, maxNumEntries);

#if defined DEBUGTIME
	auto end = Clock::now();
	float etime_0 = (end - begin).count() / 1000000;
	std::cout << "[LSH::getHash] Exit. Took " << etime_0 << "ms" << std::endl;

#endif
}

void LSH::srp_openmp_dense_data(unsigned int* hashesToFill, float *dataVal, int numInputEntries, size_t maxNumEntries) {

#pragma omp parallel for
	for (size_t inputIdx = 0; inputIdx < numInputEntries; inputIdx++) {
   		for (size_t rep = 0; rep < _numTables; rep++) {
			size_t hash = 0;
			for (size_t bit = 0; bit < _rangePow; bit++) {
				double s = 0;
				for (size_t j = 0; j < _samSize; j++) {
					size_t index = rep * _rangePow * _samSize + bit * _samSize + j;
					size_t location = _indices[index];
					double v = dataVal[_dimension * inputIdx + location];
					if (_randBits[index] >= 0) {
						s += v;
					} else {
						s -= v;
					}
				}
				hash += (s >= 0 ? 0 : 1) << bit;
			}
			hashesToFill[hashIndicesOutputIdx(_numTables, 1, maxNumEntries, inputIdx, 0, rep)] = hash;
		}
    	}
}
} //end namespace flinng