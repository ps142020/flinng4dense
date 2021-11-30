#include "io.h"

namespace flinng {

void FlinngIndex::write_index(const char* fname){
    FileIO idx_stream;
    size_t ret;
    short * bits;
    int *inds;
    std::vector<int> rambo_size;
    int lsh_size;
    idx_stream.fname = fname;
    idx_stream.fp = fopen(fname, "wb");
    if (idx_stream.fp == NULL) {
      std::cout << "Error occured during opening index file for writing\n";
      return;
    }

    //header info
    WRITE_VERIFY(&this->dimension, sizeof(this->dimension), 1, idx_stream);
    WRITE_VERIFY(&this->num_points, sizeof(this->num_points), 1, idx_stream);
    WRITE_VERIFY(&this->blooms_per_row, sizeof(this->blooms_per_row), 1, idx_stream);
    WRITE_VERIFY(&this->row_count, sizeof(this->row_count), 1, idx_stream);
    WRITE_VERIFY(&this->num_bins, sizeof(this->num_bins), 1, idx_stream);
    WRITE_VERIFY(&this->hash_repeats, sizeof(this->hash_repeats), 1, idx_stream);
    WRITE_VERIFY(&this->internal_hash_bits, sizeof(this->internal_hash_bits), 1, idx_stream);
    WRITE_VERIFY(&this->internal_hash_length, sizeof(this->internal_hash_length), 1, idx_stream);
    WRITE_VERIFY(&this->is_dataset_stored, sizeof(this->is_dataset_stored), 1, idx_stream);
    //LSH 
    WRITE_VERIFY(&this->samsize, sizeof(this->samsize), 1, idx_stream);
    this->hash_family->getLSHparams(&bits, &inds);
    lsh_size = (this->internal_hash_bits * this->hash_repeats * this->samsize);
    std::cout << "bits is: " << bits << " \n";
    WRITE_VERIFY(bits, sizeof(bits[0]), lsh_size, idx_stream);
    WRITE_VERIFY(inds, sizeof(inds[0]), lsh_size, idx_stream);

    //meta rambo
    rambo_size.resize(this->num_bins);
    this->getmetaRamboSizes(rambo_size.data());
    //write the sizes of each rambo row
    WRITE_VERIFY(rambo_size.data(), sizeof(rambo_size[0]), this->num_bins, idx_stream);
    for (uint i = 0; i < this->num_bins; i++) {
        WRITE_VERIFY(this->meta_rambo[i].data(), sizeof(this->meta_rambo[0][0]), rambo_size[i], idx_stream);
    }

    //rambo array
    rambo_size.resize(this->hash_repeats * this->internal_hash_length);
    this->getRamboSizes(rambo_size.data());
    //write the sizes of each rambo row
    WRITE_VERIFY(rambo_size.data(), sizeof(rambo_size[0]), (this->hash_repeats * this->internal_hash_length), idx_stream);
    for (uint i = 0; i < this->internal_hash_length * this->hash_repeats; i++) {
        WRITE_VERIFY(this->rambo_array[i].data(), sizeof(this->rambo_array[0][0]), rambo_size[i], idx_stream);
    }

    //bases
    if (this->is_dataset_stored) {
      WRITE_VERIFY(this->bases.data(), sizeof(this->bases[0]), (this->num_points*this->dimension), idx_stream);
    }

    fclose(idx_stream.fp);
}

void FlinngIndex::read_index(const char* fname){
    FileIO idx_stream;
    size_t ret;
    short *bits;
    int *inds;
    int *rambo_size;
    int lsh_size;
    idx_stream.fname = fname;
    idx_stream.fp = fopen(fname, "rb");
    if (idx_stream.fp == NULL) {
      std::cout << "Error occured during opening index file for reading\n";
      return;
    }

    //header info
    READ_VERIFY(&this->dimension, sizeof(this->dimension), 1, idx_stream);
    READ_VERIFY(&this->num_points, sizeof(this->num_points), 1, idx_stream);
    READ_VERIFY(&this->blooms_per_row, sizeof(this->blooms_per_row), 1, idx_stream);
    READ_VERIFY(&this->row_count, sizeof(this->row_count), 1, idx_stream);
    READ_VERIFY(&this->num_bins, sizeof(this->num_bins), 1, idx_stream);
    READ_VERIFY(&this->hash_repeats, sizeof(this->hash_repeats), 1, idx_stream);
    READ_VERIFY(&this->internal_hash_bits, sizeof(this->internal_hash_bits), 1, idx_stream);
    READ_VERIFY(&this->internal_hash_length, sizeof(this->internal_hash_length), 1, idx_stream);
    READ_VERIFY(&this->is_dataset_stored, sizeof(this->is_dataset_stored), 1, idx_stream);
    //LSH 
    READ_VERIFY(&this->samsize, sizeof(this->samsize), 1, idx_stream);
    this->hash_family = new LSH(this->internal_hash_bits,
                        this->hash_repeats, 
                        this->dimension, this->samsize, false);
    lsh_size = (this->internal_hash_bits * this->hash_repeats * this->samsize);
    this->hash_family->getLSHparams(&bits, &inds);
    std::cout << "bits now : " << bits << "\n";
      
    READ_VERIFY(bits, sizeof(bits[0]), lsh_size, idx_stream);
    READ_VERIFY(inds, sizeof(inds[0]), lsh_size, idx_stream);

    //meta rambo
    rambo_size = new int[(this->num_bins)];
    this->meta_rambo = new std::vector<uint>[this->num_bins];
    //write the sizes of each rambo row
    READ_VERIFY(rambo_size, sizeof(rambo_size[0]), (this->num_bins), idx_stream);
    for (uint i = 0; i < this->num_bins; i++) {
        std::vector<uint> rambo_buffer;
        rambo_buffer.resize(rambo_size[i]);
        READ_VERIFY(rambo_buffer.data(), sizeof(rambo_buffer[0]), rambo_size[i], idx_stream);
        this->meta_rambo[i] = rambo_buffer;
    }
    delete[] rambo_size;

    //rambo array
    rambo_size = new int[(this->hash_repeats * this->internal_hash_length)];
    this->rambo_array = new std::vector<uint32_t>[this->hash_repeats * this->internal_hash_length];
    //write the sizes of each rambo row
    READ_VERIFY(rambo_size, sizeof(rambo_size[0]), (this->hash_repeats * this->internal_hash_length), idx_stream);
    for (uint i = 0; i < this->internal_hash_length * this->hash_repeats; i++) {
        std::vector<uint32_t> rambo_buffer;
        rambo_buffer.resize(rambo_size[i]);
        READ_VERIFY(rambo_buffer.data(), sizeof(rambo_buffer[0]), rambo_size[i], idx_stream);
        this->rambo_array[i] = rambo_buffer;
    }

    //bases
    if (this->is_dataset_stored) {
      this->bases.resize(this->num_points*this->dimension);
      READ_VERIFY(this->bases.data(), sizeof(this->bases[0]), (this->num_points*this->dimension), idx_stream);
    }

    fclose(idx_stream.fp);  
    delete[] rambo_size;
}
} //end namespace flinng