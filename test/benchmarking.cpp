#include <cmath>

#include "benchmarking.h"
#include "FlinngIndex.h"
#include "utility.h"

void runBenchmark() {
  //omp_set_num_threads(1);
  omp_set_num_threads(80);

  float etime_0, etime_1, etime_2;
  auto begin = Clock::now();
  auto end = Clock::now();
  std::string idx_name = "/home/inteluser/pshidlya/similaritysearch/flinng_as_lib/test/debugDS.idx";

  std::cout << "Reading groundtruth and data ... " << std::endl;
  begin = Clock::now();

  // Read in ground truth
  unsigned int *gtruth_indice = new unsigned int[NUMQUERY * AVAILABLE_TOPK];
  readGroundTruthInt(GTRUTHINDICE, NUMQUERY, AVAILABLE_TOPK, gtruth_indice);

  // Read in queries

  float *query_val = new float[(size_t)(NUMQUERY) * DIMENSION];
	fvecs_yfcc_read_queries(QUERYFILE, DIMENSION, NUMQUERY, query_val);

  end = Clock::now();
  etime_0 = (end - begin).count() / 1000000;

  std::cout << "Read queries and GT in  " << etime_0 << "ms. \n";

  std::cout << "Reading data and indexing ... " << std::endl;
  etime_0 = 0;
//Build index
  flinng::FlinngIndex *flindex = new flinng::FlinngIndex(DIMENSION);

  size_t chunk_size = 1000000; // For now needs to be multiple of 1000000
  cout << NUMBASE << endl;
  for (size_t i = 0; i < (NUMBASE + chunk_size - 1) / chunk_size; i++) {
    size_t num_vectors = min(chunk_size, NUMBASE - i * chunk_size);
    cout << "Starting chunk " << i << ", contains " << num_vectors << " vectors." << endl;
    float *data_val_chunk = new float[(size_t)num_vectors * (size_t)DIMENSION];

    fvecs_yfcc_read_data(BASEFILE, i, num_vectors, data_val_chunk);
    
    begin = Clock::now();
    flindex->add(data_val_chunk, num_vectors);
    end = Clock::now();
    etime_0 += (end - begin).count() / 1000000;

    delete[] data_val_chunk;
 
  }

  begin = Clock::now();
  flindex->finalize_construction();
  end = Clock::now();
  etime_0 += (end - begin).count() / 1000000;

  std::cout << "Indexing " << NUMBASE << " items took " << etime_0 << "ms. \n";
  
  // verify read/write index
  flindex->write_index(idx_name.c_str());
  //delete flindex;


  //flinng::FlinngIndex *flindex_stored = new flinng::FlinngIndex(idx_name.c_str());

  //Querying
  unsigned *queryOutputs = new unsigned[NUMQUERY * TOPK]();
  float *queryDistances = new float[NUMQUERY * TOPK]();

  // Do queries
  cout << "Querying..." << endl;
  omp_set_num_threads(1);
  begin = Clock::now();
  
  //flindex_stored->search(query_val, NUMQUERY, TOPK, queryOutputs, queryDistances);
 flindex->search_with_distance(query_val, NUMQUERY, TOPK, queryOutputs, queryDistances);

  end = Clock::now();
  omp_set_num_threads(80);
  //omp_set_num_threads(1);

  etime_0 = (end - begin).count() / 1000000;
  std::cout << "Queried " << NUMQUERY << " datapoints, used " << etime_0
            << "ms. \n";

  evaluate(queryOutputs, NUMQUERY, TOPK, gtruth_indice, AVAILABLE_TOPK);

  //delete flindex;
  //delete flindex_stored;
  delete[] query_val;
  delete[] gtruth_indice;
  delete[] queryOutputs;
  delete[] queryDistances;

}

int main() {
	runBenchmark();
}
