
#pragma once

#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include "LSH.h"
#include "io.h"

using namespace std;


namespace flinng {
class FlinngBuilder {

public:
	uint row_count;
	uint blooms_per_row;
	uint num_bins;
	uint hash_repeats;
	uint internal_hash_length;
	uint internal_hash_bits;
 	int samfactor;

 	FlinngBuilder(uint rowcount = 3, uint bloomsperrow = (1<<12), 
 		      uint hash_tables = (1<<9), uint hash_table_bits = 14,
 		      int samfactor = 24) {
 		this->row_count = rowcount;
 		this->blooms_per_row = bloomsperrow;
 		this->num_bins = blooms_per_row * row_count;
 		this->hash_repeats = hash_tables;
 		this->internal_hash_bits = hash_table_bits;
 		this->internal_hash_length = 1 << internal_hash_bits;
 		this->samfactor = samfactor;
 
        }
 	~FlinngBuilder();
};

class FlinngIndex {

private:
	
    	std::vector<float> bases; /// database vectors, size ntotal * dimension
	std::vector<uint32_t> *rambo_array; //for now only adding support for 32bit
	std::vector<uint> *meta_rambo;
	std::vector<uint> hashes; 
	uint num_points; //total nb of indexed vectors
	LSH *hash_family;
	uint hash_offset;
	uint dimension;	
	uint row_count;
	uint blooms_per_row;
	uint num_bins;
	uint hash_repeats;
	uint internal_hash_length;
	uint internal_hash_bits;
 	int samfactor;
 	int samsize;
 	bool is_dataset_stored;
 	std::vector<uint> get_hashed_row_indices(uint index);

public:
	FlinngIndex(uint dimension, FlinngBuilder *def = NULL);
	FlinngIndex(const char* fname);
	FlinngIndex();
	~FlinngIndex();
	void finalize_construction();
	void add(float *x, uint num);
	void add_and_store(float *input, uint num_items);
	void search(float* query, unsigned n, unsigned k, unsigned* descriptors);
	void search_with_distance(float *queries, unsigned num_queries, unsigned topk,
                                  unsigned* descriptors, float* distances);
	void write_index(const char* fname);
	void read_index(const char* fname);
	void fetch_descriptors(long id, float* desc);
	void getRamboSizes(int *sizes);
	void getmetaRamboSizes(int *sizes);
};
} //end namespace flinng
