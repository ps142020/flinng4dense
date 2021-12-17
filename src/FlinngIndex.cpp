#include "FlinngIndex.h"
#include "MurmurHash3.h"


//#define DEBUG
//#define DEBUGTIME

using namespace std;
typedef std::chrono::high_resolution_clock Clock;
namespace flinng {

vector<uint> FlinngIndex::get_hashed_row_indices(uint index) {
  string key = to_string(index);
  uint key_length = to_string(index).size();
  vector<uint> hashvals(0);
  uint op;
  for (uint i = 0; i < this->row_count; i++) {
    MurmurHash3_x86_32(key.c_str(), key_length, i, &op); // seed is row number
    hashvals.push_back(op % blooms_per_row);
  }
  return hashvals;
}

FlinngIndex::FlinngIndex(uint dimension, FlinngBuilder *def) {
	if (def == NULL) {
		def = new FlinngBuilder();
	}
  this->dimension = dimension;
	this->hash_offset = 0;
  this->num_points = 0;
  this->samfactor = def->samfactor;
  this->samsize = (int) floor(this->dimension / this->samfactor);
	this->row_count = def->row_count;
	this->blooms_per_row = def->blooms_per_row;
	this->num_bins = this->blooms_per_row * this->row_count;
	this->hash_repeats = def->hash_repeats;
	this->internal_hash_bits = def->internal_hash_bits;
	this->internal_hash_length = 1 << this->internal_hash_bits;
  this->rambo_array = new vector<uint32_t>[hash_repeats * internal_hash_length];
	this->hash_family = new LSH(this->internal_hash_bits,
		                    this->hash_repeats, 
		                    this->dimension, this->samsize, true);
}

FlinngIndex::FlinngIndex(const char* fname) {

  this->read_index(fname);
}

FlinngIndex::~FlinngIndex() {

  delete[] meta_rambo;
  delete[] rambo_array;

}

/**
 * Finishes FLINNG construction by sorting all buckets for fast access. All
 * points must be inserted at this point.
 */
void FlinngIndex::finalize_construction() {

#if defined DEBUGTIME
  std::cout << "[FlinngIndex::finalize_construction] " << std::endl;
  auto begin = Clock::now();
#endif

//By now all insertions are done, so we prepare meta rambo for searching
  cout << "Getting row indices" << endl;
  vector<vector<uint>> *row_indices_arr = new vector<vector<uint>>(this->num_points);
#pragma omp parallel for
  for (uint i = 0; i < this->num_points; i++) {
    row_indices_arr->at(i) = get_hashed_row_indices(i);
  }

  cout << "Creating meta rambo" << endl;
  // Create meta rambo
  meta_rambo = new vector<uint>[this->row_count * this->blooms_per_row];
  for (uint point = 0; point < this->num_points; point++) {
    vector<uint> hashvals = row_indices_arr->at(point);
    for (uint r = 0; r < this->row_count; r++) {
      this->meta_rambo[hashvals.at(r) + this->blooms_per_row * r].push_back(point);
    }
  }

  cout << "Sorting meta rambo" << endl;
  // Sort array entries in meta rambo
#pragma omp parallel for  
  for (uint i = 0; i < this->num_bins; i++) {
    sort(this->meta_rambo[i].begin(), this->meta_rambo[i].end());
  }

  row_indices_arr->clear();
  delete row_indices_arr;


//insert into rambo array
  cout << "Getting row indices" << endl;
  vector<vector<uint>> *row_indices_arr2 = new vector<vector<uint>>(this->num_points);
#pragma omp parallel for
  for (uint i = 0; i < this->num_points; i++) {
    row_indices_arr2->at(i) = get_hashed_row_indices(i);
  }

  cout << "Populating FLINNG" << endl;
#pragma omp parallel for
  for (uint rep = 0; rep < this->hash_repeats; rep++) {
    for (uint index = 0; index < this->num_points; index++) {
      vector<uint> row_indices = row_indices_arr2->at(index);
      for (uint r = 0; r < this->row_count; r++) {
        uint b = row_indices.at(r);
        this->rambo_array[rep * internal_hash_length +
                    this->hashes[index  * this->hash_repeats + rep]]
            .push_back(r * this->blooms_per_row + b);
      }
    }
  }

  row_indices_arr2->clear();
  delete row_indices_arr2;

  // Remove duplicates
  cout << "Sorting FLINNG" << endl;
#pragma omp parallel for
  for (uint i = 0; i < this->internal_hash_length * this->hash_repeats; i++) {
    sort(this->rambo_array[i].begin(), this->rambo_array[i].end());
    this->rambo_array[i].erase(unique(this->rambo_array[i].begin(), this->rambo_array[i].end()),
                         this->rambo_array[i].end());
  }

#if defined DEBUGTIME  
  auto end = Clock::now();
  float etime_0 = (end - begin).count() / 1000000;
  std::cout << "[FlinngIndex::finalize_construction] exit. Took " << etime_0 << "ms" << std::endl;

#endif

}

void FlinngIndex::getRamboSizes(int *sizes) {
  for (uint i = 0; i < this->internal_hash_length * this->hash_repeats; i++) {
    sizes[i] = this->rambo_array[i].size();
  }
}

void FlinngIndex::getmetaRamboSizes(int *sizes) {
  for (uint i = 0; i < this->num_bins; i++) {
    sizes[i] = this->meta_rambo[i].size();
  }
}

/**
 * Inserts a keys of a given index into the FLINNG array
 */
void FlinngIndex::add(float *input, uint num_items) {

#if defined DEBUGTIME
  auto begin = Clock::now();
#endif

this->hashes.resize((this->num_points + num_items) * this->hash_repeats);

#if defined DEBUGTIME  
  auto end = Clock::now();
  float etime_0 = (end - begin).count() / 1000000;
  std::cout << "FlinngIndex::add, resizing hashes took " << etime_0 << "ms" << std::endl;
#endif

this->hash_family->getHash(this->hashes.data(), input, num_items, this->num_points);
this->num_points += num_items;
this->is_dataset_stored = false;

}

/**
 * Inserts a keys of a given index into the FLINNG array and stores the original dataset
 */
void FlinngIndex::add_and_store(float *input, uint num_items) {

  add(input, num_items);

#if defined DEBUGTIME
  auto begin = Clock::now();
#endif
  
  this->bases.insert(this->bases.end(), input, input + num_items * dimension);
  this->is_dataset_stored = true;

#if defined DEBUGTIME  
  auto end = Clock::now();
  float etime_0 = (end - begin).count() / 1000000;
  std::cout << "FlinngIndex::add_and_store, inserts into bases took " << etime_0 << "ms" << std::endl;
#endif
}

static float cosineDist(float *A, float *B, unsigned int n) {

  float up = 0;
  for (int i = 0; i < n; i++) up += A[i] * B[i];

  float a = 0;
  for (int i = 0; i < n; i++) a += A[i] * A[i];

  float b = 0;
  for (int i = 0; i < n; i++) b += B[i] * B[i];
  a = sqrtf(a);
  b = sqrtf(b);

  return up / (a * b);
}


void FlinngIndex::search(float *queries, unsigned num_queries, unsigned topk, unsigned* descriptors) {
  
unsigned int *query_hashes = new unsigned int[this->hash_repeats];

#if defined DEBUGTIME
  std::cout << "[FlinngIndex::search] " << std::endl;
  auto begin = Clock::now();
  auto end = Clock::now();
  float etime_0;

#endif

  for (int p = 0; p < num_queries ; p++) {
    memset(query_hashes, 0, this->hash_repeats*sizeof(*query_hashes));
    this->hash_family->getHash(query_hashes, queries + this->dimension * p, 1, 0);

    // Get observations, ~80%!
    vector<uint> counts(this->num_bins, 0);
    for (uint rep = 0; rep < this->hash_repeats; rep++) {
      const uint index = this->internal_hash_length * rep + query_hashes[rep];
      const uint size = this->rambo_array[index].size();
      for (uint small_index = 0; small_index < size; small_index++) {
        // This single line takes 80% of the time, around half for the move and
        // half for the add
        ++counts[this->rambo_array[index][small_index]];
      }
    }

    vector<uint> sorted[hash_repeats + 1];
    uint size_guess = num_bins / (hash_repeats + 1);
    for (vector<uint> &v : sorted) {
      v.reserve(size_guess);
    }
    for (uint i = 0; i < num_bins; ++i) {
      sorted[counts[i]].push_back(i);
    }

    if (row_count > 2 || num_points < 4000000)  {
      vector<uint8_t> num_counts(num_points, 0);
      uint num_found = 0;
      for (int rep = hash_repeats; rep >= 0; --rep) {
        for (uint bin : sorted[rep]) {
          for (uint point : meta_rambo[bin]) {
            if (++num_counts[point] == row_count) {
              descriptors[p * topk + num_found] = point;
              if (++num_found == topk) {
                // cout << "Using threshhold " << rep << endl;
                //return;
                goto next_query;
              }
            }
          }
        }
      }
    }
    else {
      char* num_counts = (char*) calloc(num_points / 8 + 1, sizeof(char));
      uint num_found = 0;
      for (int rep = hash_repeats; rep >= 0; --rep) {
        for (uint bin : sorted[rep]) {
          for (uint point : meta_rambo[bin]) {
            if (num_counts[(point / 8)] & (1 << (point % 8))) {
              descriptors[p * topk + num_found] = point;
              if (++num_found == topk) {
                // cout << "Using threshhold " << rep << endl;
                free(num_counts);
                //return;
                goto next_query;
              }
            } else {
              num_counts[(point / 8)] |= (1 << (point % 8));
            }
          }
        }
      }
    }
  next_query: /* done with one query*/;
  } //end of all queries

#if defined DEBUGTIME  
  end = Clock::now();
  etime_0 = (end - begin).count() / 1000000;
  std::cout << "[FlinngIndex::search] exit. Took " << etime_0 << "ms" << std::endl;
#endif

  delete[] query_hashes;

}

void FlinngIndex::search_with_distance(float *queries, unsigned num_queries, unsigned topk,
                                       unsigned* descriptors, float* distances) {

  search(queries, num_queries, topk, descriptors);

  if (!this->is_dataset_stored) {
    std::cout << "Dataset is not stored! Distance cannot be calculated. Invoke add_with_store() to store dataset. \n";
    return;
  }

#if defined DEBUGTIME  
   float etime_0;
  std::cout << "[FlinngIndex::search_with_distance]  " << std::endl;
  //begin = Clock::now();
  auto begin = std::chrono::high_resolution_clock::now();
#endif

  float dist;
#pragma omp parallel private(dist)
#pragma omp parallel for
  /* Output distances. */
  for (int i = 0; i < num_queries; i++) {
    for (int j = 0; j < topk; j++) {

       dist = cosineDist(queries + dimension * i, bases.data() + dimension * descriptors[i * topk + j], dimension);
       distances[i * topk + j] = dist;
    }
  }

#if defined DEBUGTIME  
  auto end = Clock::now();
  etime_0 = (end - begin).count() / 1000000;

  std::cout << "[FlinngIndex::search_with_distance] finding distances took " << etime_0 << "ms" << std::endl;
#endif

}

void FlinngIndex::fetch_descriptors(long id, float* desc){
#if defined DEBUGTIME  
  std::cout << "[FlinngIndex::fetch_descriptors]  " << std::endl;
  auto begin = Clock::now();
#endif
    memcpy(desc, &(bases[id * dimension]), sizeof(*desc) * dimension);

#if defined DEBUGTIME  
  float etime_0;
  auto end = Clock::now();
  etime_0 = (end - begin).count() / 1000000;

  std::cout << "[FlinngIndex::fetch_descriptors] exit. Took " << etime_0 << "ms" << std::endl;
#endif

}

} //end namespace flinng
