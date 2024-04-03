
#include <__clang_cuda_builtin_vars.h>
#include <iostream>
#include "cuda_runtime.h"



struct DD {
  int *m_xs;
  Double *m_ps;
  size_t m_size;


  DD(int *xs, double *ps, size_t size){
    this->m_xs = xs;
    this->m_ps = ps;
    this->m_size = size;
  }

  void print(){
    std::cout << "Size: " << m_size << std::endl;
    for (int i = 0; i < m_size; i++){
	std::cout << m_xs[i] << " " << m_ps[i] << std::endl;
    }
  }
};



__global__ void convolution_kernel(float *p1, float *p2,  float *p3, int size){

  p3[threadIdx.x] = p1[threadIdx.x] * p2[threadIdx.x];

  // std::cout << "Convolution Kernel\n";
  
}

DD convolution(float *p1, float *p2, int size){
  std::cout << "Convolution" << std::endl;
  return DD(NULL, NULL, 0);
}


int main(){
  int xs[] = {1, 2, 3, 4, 5};
  double ps[] = {.2, .2, .2, .2, .2};
  DD d1(xs, ps, 5);
  DD d2(xs, ps, 5);

  float p1[] = {1, 2, 3, 4, 5};
  float p2[] = {1, 2, 3, 4, 5};
  float p3[] = {0, 0, 0, 0, 0};
  d1.print();
  //convolution(d1, d2);
  
  float *d_p1, *d_p2, *d_p3;

  cudaMalloc(&d_p1, 5 * sizeof(float));
  cudaMalloc(&d_p2, 5 * sizeof(float));
  cudaMalloc(&d_p3, 5 * sizeof(float));

  cudaMemcpy(d_p1, p1, 5 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p2, p2, 5 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p3, p3, 5 * sizeof(float), cudaMemcpyHostToDevice);

  convolution_kernel<<<1, 1>>>(d_p1, d_p2, d_p3, 5);

  cudaMemcpy(p3, d_p3, 5 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++){
    std::cout << p3[i] << std::endl;
  }

}

