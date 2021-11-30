#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <string>
#include <fstream>

using namespace std;

class BinaryReader {

    std::ifstream _ifs;

    uint32_t _dim;

    std::string _prefix;
    int _file_counter;
    long _read_counter;

public:
    BinaryReader(uint32_t dim, std::string prefix);
    BinaryReader(uint32_t dim, int file_num, std::string prefix);

    void open(std::string filename);
    int read(int vectors, float* buff, size_t buf_size,
                     long* ids, size_t id_buf_size);
};

void readGroundTruthInt(const std::string& file, int numQueries, int availableTopK, unsigned int *out);
void readGroundTruthFloat(const std::string& file, int numQueries, int availableTopK, float *out);

void fvecs_yfcc_read_data(const std::string& file_prefix, int offset, int readsize, float* start);
void fvecs_yfcc_read_queries(const std::string& file, int dim, int readsize, float* out);

void evaluate(
	unsigned int *queryOutputs,		// The output indices of queries.
	int numQueries,			// The number of query entries, should be the same for outputs and groundtruths.
	int topk,				// The topk per query contained in the queryOutputs.
	unsigned int *groundTruthIdx,	// The groundtruth indice vector.
	int availableTopk		// Available topk information in the groundtruth.
	);

void rMetric(unsigned int *queryOutputs, int numQueries, int topk,
	unsigned int *groundTruthIdx, int availableTopk, int numerator);

