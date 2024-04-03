#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>
#include <random>
#include <vector_types.h>



struct DD {

  int *m_xs;
  double *m_ps;
  size_t m_size;

  DD(int *xs, double *ps, size_t size) {
    this->m_xs = xs;
    this->m_ps = ps;
    this->m_size = size;
  }

  void print() {
    std::cout << "Size: " << m_size << std::endl;
    for (int i = 0; i < m_size; i++) {
      std::cout << m_xs[i] << " " << m_ps[i] << std::endl;
    }
  }
};

void print_array(float *ar, int len){

  for (int i = 0; i < len; i++){
    std::cout << "ar @ pos " << i << " = " << ar[i] << "\n";
  }
}

__global__ void convolution_kernel(float *p1, float *p2, float *p3, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  //p3[i] = p1[i] + p2[i];
  if (i < size && j < size){
    p3[i * size + j] = p1[i] * p2[j];
  }

  //int i = threadIdx.x;
  //int j = threadIdx.y;
  //if (threadIdx.x <= size &&  threadIdx.y <= size) {
  //if (i <= size &&  j <= size) {
    //p3[threadIdx.x * size + threadIdx.y] = p1[threadIdx.x] * p2[threadIdx.y];
  //p3[i * size + j] = p1[i] * p2[j];
  //p3[threadIdx.x * size + threadIdx.y] = p1[threadIdx.x] * p2[threadIdx.y];
  //	}
}


__global__ void add_kernel(float *p1, float *p2, float *p3, int size){
  int i = threadIdx.x;
  p3[i] = p1[i] + p2[i];
}

DD convolution(float *p1, float *p2, int size) {
  std::cout << "Convolution" << std::endl;
  return DD(NULL, NULL, 0);
}


void populate_with_randvalues(float *ar, int len){
  float sum = 0.0;
  std::mt19937 r_gen{std::random_device{}()};
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < len; i++) {
    ar[i] = (float) dist(r_gen);
    sum += ar[i];
  }
  for (int i = 0; i < len; i++) {
    ar[i] = ar[i] / sum;
  }
  std::cout << "Sum: " << sum << std::endl;
}


template <typename T>
int check_alloc(T *ptr){
  if (ptr == NULL){
    std::cout << "Error allocating memory\n";
    return -1;
	}
  else{
    std::cout << "Memory allocated successfully.\n";
    return 1;
  }



}

int main() {


  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "Device count: " << deviceCount << std::endl;



  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device name: " << prop.name << std::endl;
  std::cout << "Memory: " << prop.totalGlobalMem << std::endl;
  std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << std::endl;
  std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
  std::cout << "Warp size: " << prop.warpSize << std::endl;
  std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Max threads dim: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;

  size_t SIZE = 100;
  size_t inp_size_alloc = SIZE * sizeof(float);
  size_t res_size_alloc = SIZE * SIZE * sizeof(float);

  std::mt19937 r_gen{};
  
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < 10; i++) {
		std::cout << dist(r_gen) << std::endl;
	}


  //int xs[] = {1, 2, 3, 4, 5};
  //double ps[] = {.2, .2, .2, .2, .2};
  //DD d1(xs, ps, 5);
  //DD d2(xs, ps, 5);


  float *p1 = (float *) malloc(inp_size_alloc);
  float *p2 = (float *) malloc(inp_size_alloc);
  //float p1[SIZE] = {0.0};
  check_alloc<float>(p1);
  //float p2[SIZE] = {0.0};
  check_alloc<float>(p2);
  float *p3 = (float *) malloc(res_size_alloc);
  check_alloc<float>(p3);


  populate_with_randvalues(p1, SIZE);
  populate_with_randvalues(p2, SIZE);
  print_array(p1, SIZE);
  print_array(p2, SIZE);

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  std::cout << "Free memory: " << free_mem << std::endl;
  std::cout << "Total memory: " << total_mem << std::endl;

  float *d_p1;
  cudaMalloc(&d_p1, inp_size_alloc);
  std::cout << "Allocated memory on device for p1\n";


  float *d_p2;
  cudaMalloc(&d_p2, inp_size_alloc);
  std::cout << "Allocated memory on device for p2\n";

  float *d_p3; 
  cudaMalloc(&d_p3, res_size_alloc);

  float *d_p4; 
  cudaMalloc(&d_p4, inp_size_alloc);

  std::cout << "Allocated memory on device for p3\n";

  cudaMemcpy(d_p1, p1, inp_size_alloc, cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2, p2, inp_size_alloc, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_p3, p3, res_size_alloc, cudaMemcpyHostToDevice);



  //int num_threads = 512;
  int num_threads = 20;
  int num_blocks = 5;
  //int num_blocks = std::ceil(SIZE / num_threads);
  dim3 ts(num_threads, num_threads, 1);
  dim3 bs(num_blocks, num_blocks, 1);
  std::cout << "Num blocks: " << num_blocks << std::endl;
  std::cout << "Num threads: " << num_threads << std::endl;
  
  convolution_kernel<<<1, ts>>>(d_p1, d_p2, d_p3, num_threads);
  //convolution_kernel<<<1, num_threads>>>(d_p1, d_p2, d_p3, SIZE);

  //convolution_kernel<<<bs, ts>>>(d_p1, d_p2, d_p3, num_threads);
  
  // print_array(p3, 10);
  cudaMemcpy(p3, d_p3, res_size_alloc, cudaMemcpyDeviceToHost);
  print_array(p3, 100);

  float *p4 = (float *) malloc(SIZE * sizeof(float));
  *p4 = 0.0;
  add_kernel<<<1, num_threads>>>(d_p1, d_p2, d_p4, num_threads);
  cudaMemcpy(p4, d_p4, inp_size_alloc, cudaMemcpyDeviceToHost);

  //print_array(p3, SIZE*SIZE);
  //print_array(p4, 10);
  // cudaFree(d_p1);
  //cudaFree(d_p2);
  //cudaFree(d_p3);
  //delete p3;
  //print_array(p1, SIZE);
  //print_array(p2, SIZE);

}
