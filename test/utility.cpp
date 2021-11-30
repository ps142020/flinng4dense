#include <sstream>
#include <vector>
#include <math.h>
#include <iostream>
#include <assert.h>
#include "benchmarking.h"
#include "utility.h"

/*
* Function:  evaluate
* --------------------
* Evaluate the results of a dataset using various metrics, prints the result
*
*  returns: nothing
*/
void evaluate (
	unsigned int *queryOutputs,		// The output indices of queries.
	int numQueries,			// The number of query entries, should be the same for outputs and groundtruths.
	int topk,				// The topk per query contained in the queryOutputs.
	unsigned int *groundTruthIdx,	// The groundtruth indice vector.
	int availableTopk		// Available topk information in the groundtruth.
	) {				// The number of n(s) interested.

	rMetric(queryOutputs, numQueries, topk, groundTruthIdx, availableTopk, 1);
	rMetric(queryOutputs, numQueries, topk, groundTruthIdx, availableTopk, 10);
	if (availableTopk >= 100) {
		rMetric(queryOutputs, numQueries, topk, groundTruthIdx, availableTopk, 100);
	} else {
		rMetric(queryOutputs, numQueries, topk, groundTruthIdx, availableTopk, availableTopk);
	}
	
}

void rMetric(unsigned int *queryOutputs, int numQueries, int topk,
	unsigned int *groundTruthIdx, int availableTopk, int numerator) {

        printf("Top 10 neighbours of 5 query\n");
        for (int q=0; q<5; q++) {
                printf("Query: %d ", q);
                for (int i = 0; i < 10; i++) {
                        printf("%d ", queryOutputs[q*topk + i]);
                }
                printf("\n");
        }

	printf("\nR10@k Average fraction of top %d nearest neighbors returned in k first results. \n", numerator);

	int *good_counts = new int[topk]();

	for (int i = 0; i < numQueries; i++) {
		unordered_set<unsigned int> topGtruths(groundTruthIdx + i * availableTopk, groundTruthIdx + i * availableTopk + numerator);
		for (int denominator = numerator; denominator <= topk; denominator++) {
			unordered_set<unsigned int> topOutputs(queryOutputs + i * topk, queryOutputs + i * topk + denominator);
			uint local_count = 0;
			for (const auto& elem : topGtruths) {
				if (topOutputs.find(elem) != topOutputs.end()) { // If elem is found in the intersection.
					local_count++;
					good_counts[denominator - numerator]++;
				}
			}
		}
	}

	for (int denominator = numerator; denominator <= topk; denominator++) {
		printf("R%d@%d = %1.3f \n", numerator, denominator, (float)good_counts[denominator - numerator] / numQueries / numerator);
	}
	printf("\n"); printf("\n");

	delete[] good_counts;

}

//* Functions for reading GT, query file and dataset *//

BinaryReader::BinaryReader(uint32_t dim, std::string prefix):
    _prefix(prefix),
    _dim(dim),
    _file_counter(0),
    _read_counter(0)
{
    std::string filename = _prefix + std::to_string(_file_counter) + ".bin";
    open(filename);
}

BinaryReader::BinaryReader(uint32_t dim, int file_num, std::string prefix):
    _prefix(prefix),
    _dim(dim),
    _file_counter(file_num),
    _read_counter(0)
{
    std::string filename = _prefix + std::to_string(_file_counter) + ".bin";
    open(filename);
}

void BinaryReader::open(std::string filename)
{
    if (_ifs.is_open())
        _ifs.close();

    std::cout << "Opening " << filename << std::endl;
    _ifs.open(filename);

    if (!_ifs)
        printf("problem opening file: filename\n");
}

int BinaryReader::read(int vectors,
                       float* vbuff, size_t buf_size,
                       long* idbuff, size_t ids_buf_size)
{
    if (buf_size < vectors * _dim) {
        printf("not enough space in buffer features\n");
    }

    if (ids_buf_size < vectors ) {
        printf("not enough space in buffer ids\n");
    }

    char str[256];

    for (int i = 0; i < vectors; ++i) {

        if (_read_counter % int(1e6) == 0 && _read_counter != 0) {
            _file_counter++;
            std::string filename = _prefix + std::to_string(_file_counter) + ".bin";
            open(filename);
        }

        if (_ifs.eof()) {
            return i;
        }
        _ifs.read((char*)&(idbuff[i]), sizeof(long)); // read id
        _ifs.read((char*)&(vbuff[i*_dim]), sizeof(float) * _dim);

        _read_counter++;

    }

    return vectors;
}

void compute_averages() {
  cout << "Computing averages" << endl;
  int batch = 1000;
  float* fvs = new float[DIMENSION * batch];
  long* ids  = new long[batch];        // not used

  BinaryReader reader(DIMENSION, BASEFILE);
  size_t features_read = 0;
  double totals[DIMENSION + 1] = {};
  while (features_read < NUMBASE) {
        size_t read = reader.read(batch, fvs, DIMENSION*batch, ids, batch);
	if (read == 0) {
	  cout << features_read << endl;
	}
#pragma omp parallel for
        for (size_t d = 0; d < DIMENSION; d++) {
          for (size_t i = 0; i < read; i++) {
            totals[d] += fvs[i * DIMENSION + d];
          }
        }


#pragma omp parallel for
        for (size_t i = 0; i < read; i++) {
          double magnitude = 0;
          for (size_t d = 0; d < DIMENSION; d++) {
            float component = fvs[i * DIMENSION + d];
            magnitude += component * component;
          }
#pragma omp critical        
          totals[DIMENSION] = max(totals[DIMENSION], sqrt(magnitude));
        }
      	features_read += read;
  }

  for (size_t i = 0; i < DIMENSION; i++) {
    totals[i] /= NUMBASE;
  }

  float averages[DIMENSION + 1];
  for (uint i = 0; i <= DIMENSION; i++) {
    averages[i] = totals[i];
  }

  FILE * pFile;
  pFile = fopen (("averages" + to_string(NUMBASE) + ".bin").c_str(), "wb");
  fwrite(averages, sizeof(float), DIMENSION + 1, pFile);
  fclose(pFile);
}

void get_averages(float *buffer) {
  cout << "Getting averages" << endl;
  FILE *pFile = fopen (("averages" + to_string(NUMBASE) + ".bin").c_str(), "rb");
  if (!pFile) {
    compute_averages();
    get_averages(buffer);
    return;
  }
  size_t ret = fread(buffer, sizeof(float), DIMENSION + 1, pFile);
  fclose(pFile);
}

/* Functions for reading and parsing the YFCC100M dataset. */

void fvecs_yfcc_read_data(const std::string& file_prefix, int offset, int readsize, float* start) {
    int batch = 1000;
    size_t index = 0;

    float* fvs = new float[DIMENSION * batch];
    long* ids  = new long[batch];        // not used

    float averages[DIMENSION + 1];
    get_averages(averages);

    if (offset > 97 || offset < 0) {
        printf("offset needs to be a valid file number for YFCC100M... \n");
        exit(EXIT_FAILURE);
    }

    BinaryReader reader(DIMENSION, offset, file_prefix);

    size_t features_read = 0;
    while (features_read < readsize) {
        size_t read = reader.read(batch, fvs, DIMENSION * batch, ids, batch);
#pragma omp parallel for
        for (size_t i = 0; i < read * DIMENSION; i++) {
          start[index + i] = (fvs[i] - averages[i % DIMENSION]) / averages[DIMENSION];
        }
        index += read * DIMENSION;
      	features_read += read;
    }

}

void fvecs_yfcc_read_queries(const std::string& file, int dim, int readsize, float* out) {
  ifstream in(file);
  string line;
  float averages[DIMENSION + 1];
  get_averages(averages);
  for (int line_num = 0; line_num < readsize; line_num++) {
    getline(in, line);
    stringstream ss(line);
    string buff;
    for (int d = 0; d < dim; d++){
      getline(ss, buff, ' ');
      out[line_num * dim + d] = (stof(buff) - averages[d]) / averages[DIMENSION];
    }
  }
  // partly courtesy of:
  // https://stackoverflow.com/questions/1894886/parsing-a-comma-delimited-stdstring

}


/*
For reading the indices of the groudtruths.
Each indice represent a datapoint in the same order as the base dataset.
Vector indexing:
The k_th neighbor of the q_th query is out[(q * availableTopK)]
file - filename
numQueries - the number of query data points
availableTopK - the topk groundtruth available for each vector
out - output vector
*/
void readGroundTruthInt(const std::string &file, int numQueries,
                        int availableTopK, unsigned int *out) {
  std::ifstream myFile(file, std::ios::in | std::ios::binary);

  if (!myFile) {
    printf("Error opening file ... \n");
    return;
  }

  char cNum[256];
  int ct = 0;
  while (myFile.good() && ct < availableTopK * numQueries) {
    myFile.good();
    myFile.getline(cNum, 256, ' ');
    out[ct] = atoi(cNum);
    ct++;
  }

  myFile.close();
}

/*
For reading the distances of the groudtruths.
Each distances represent the distance of the respective base vector in the
"indices" to the query. Vector indexing: The k_th neighbor's distance to the
q_th query is out[(q * availableTopK) + k] file - filename numQueries - the
number of query data points availableTopK - the topk groundtruth available for
each vector out - output vector
*/
void readGroundTruthFloat(const std::string &file, int numQueries,
                          int availableTopK, float *out) {
  std::ifstream myFile(file, std::ios::in | std::ios::binary);

  if (!myFile) {
    printf("Error opening file ... \n");
    return;
  }

  char cNum[256];
  int ct = 0;
  while (myFile.good() && ct < availableTopK * numQueries) {
    myFile.good();
    myFile.getline(cNum, 256, ' ');
    out[ct] = strtof(cNum, NULL);
    ct++;
  }

  myFile.close();
}


