
#include <iostream>


struct DD {
  int *m_xs;
  double *m_ps;
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




DD convolution(DD& d1, DD& d2){
  std::cout << "Convolution" << std::endl;
  return DD(NULL, NULL, 0);
}


int main(){
  int xs[] = {1, 2, 3, 4, 5};
  double ps[] = {.2, .2, .2, .2, .2};
  DD d1(xs, ps, 5);
  DD d2(xs, ps, 5);
  d1.print();
  convolution(d1, d2);
}

