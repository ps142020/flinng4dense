#pragma once
#include <algorithm>    // std::random_shuffle
#include <chrono>
#include <cstdlib>      // std::rand, std::srand
#include <ctime>        // std::time
#include <iostream>
#include <random>
#include "omp.h"

namespace flinng {

#define hashIndicesOutputIdx(numTables, numProbes, numInputs, dataIdx, probeIdx, tb) (unsigned)((numInputs + dataIdx) * numProbes * numTables + tb * numProbes + probeIdx)

class LSH {
private:

	/* Core parameters. */
	int _rangePow;
	int _numTables;
	int _dimension;

	/* Signed random projection. */
	int _samSize;
	short *_randBits;
	int *_indices;
	/* Function definitions. */
	void srp_openmp_dense_data(unsigned int *hashes, float *dataVal, int numInputEntries, size_t maxNumEntries);


public:
	/** Obtain hash indice given the input vector.
	Hash indice refer to the corresponding "row number" in a hash table, in the form of unsigned integer.
	The outputs indexing is defined as macro hashIndicesOutputIdx 
	@param hashIndices Hash indice for the batch of input vectors.
	@param dataVal Non-zero values of the dense format.
	@param numInputEntries Number of input vectors in current invocation as pointed by dataVal.
	@param maxNumEntries Number of input vectors already invoked so far.
	*/
	void getHash(unsigned int *hashes, float *dataVal, int numInputEntries, size_t maxNumEntries);

	/** Constructor.
	Construct an LSH class for signed random projection.
	@param numHashPerFamily Number of hash (bits) per hashfamily (hash table).
	@param numHashFamilies Number of hash families (hash tables).
	@param dimension Dimensionality of input data.
	@param samFactor samFactor = dimension / samSize, have to be an integer.
	*/
	LSH(int numHashPerFamily, int numHashFamilies, int dimension, int samsize, bool init);
	~LSH();
	void getLSHparams(short **bits, int **indices);
};
} //end namespace flinng