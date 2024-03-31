from tensor import Tensor, TensorSpec, TensorShape
from random import randn, rand
from sys.info import simdwidthof
from algorithm import vectorize
alias simd_width: Int = simdwidthof[tpe]()
from math import add
from python import Python


alias type = DType.float32
alias tpe = DType.float64



struct TensorTuple:
  var t1: Tensor[tpe]
  var t2: Tensor[tpe]

  fn __init__(inout self, first_t: Tensor[tpe], second_t: Tensor[tpe]):
    self.t1 = first_t
    self.t2 = second_t

  fn __copyinit__(inout self, existing: TensorTuple) -> None:
    self.t1 = self.t1
    self.t2 = self.t2

  fn __repr__(inout self) -> String:
    return String(self.t1) + String(self.t2)


fn convolution(distr1: TensorTuple, distr2: TensorTuple) -> TensorTuple:
    var res_values: Tensor[tpe]
    var res_probs: Tensor[tpe]
    res_values = outer_sum(distr1.t1, distr2.t1)
    res_probs =outer_product(distr1.t2, distr2.t2)
    return TensorTuple(res_values, res_probs)


fn outer_sum(t1: Tensor[tpe], t2: Tensor[tpe]) -> Tensor[tpe]:
    let s1: Int = t1.num_elements()
    let s2: Int = t2.num_elements()
    var t3: Tensor[tpe] = Tensor[tpe](s1 * s2)

    for i in range(t1.num_elements()):
      for j in range(t2.num_elements()):
          t3[i * s2 +j] = t1[i] + t2[j]
    return t3


fn outer_product(t1: Tensor[tpe], t2: Tensor[tpe]) -> Tensor[tpe]:
    let s1: Int = t1.num_elements()
    let s2: Int = t2.num_elements()
    var t3: Tensor[tpe] = Tensor[tpe](s1 * s2)
    for i in range(t1.num_elements()):
      for j in range(t2.num_elements()):
          t3[i * s2 + j] = t1[i] * t2[j]
    return t3

fn outer_product_vec(t1: Tensor[tpe], t2: Tensor[tpe]) -> Tensor[tpe]:
    let s1: Int = t1.num_elements()
    let s2: Int = t2.num_elements()
    var t3: Tensor[tpe] = Tensor[tpe](s1 * s2)

    var j: Int = 0


    print(s2)
    for j in range(s2):


      print("j = ", j)
      @parameter
      fn vec_outer_product[simd_width: Int](idx_1: Int) -> None:
        print("t2 value at j = ", j, " is ", t2[j])
        t3.simd_store[simd_width](s2 * idx_1 + j , t1.simd_load[simd_width](idx_1) * t2.simd_load[simd_width](j))
        # t3.simd_store[simd_width](((s2 * idx_1) + j) , t1.simd_load[simd_width](idx_1) * t2[j])
        #t3.simd_store[simd_width](idx_1, t1.simd_load[simd_width](idx_1) + t2.simd_load[simd_width](idx_1))
          
        print("j =" + String(j))
        print(idx_1)
      vectorize[simd_width, vec_outer_product](s1)


    return t3


fn create_tensor(*vals: Float64) -> Tensor[tpe]:
     var t: Tensor[tpe] = Tensor[tpe](len(vals))
     for i in range(len(vals)):
         t[i] = vals[i]
     return t
     
fn py_test() raises:
    Python.add_to_path(".")
    # var py_code = Python.import_module("preliminary_tests")
    var np = Python.import_module("numpy")
    var ar = np.array([1,2,3])
    var ar_1: PythonObject = np.random.random(100000)
    var ar_2: PythonObject = np.random.random(100000)
    var ar_3: PythonObject = np.add(ar_1, ar_2).flatten()
    var ar_4: PythonObject = np.multiply(ar_1, ar_2).flatten()

fn benchmark_py() raises:
    var np = Python.import_module("numpy")
    var ar_1: PythonObject = np.random.random(100000)
    var ar_2: PythonObject = np.random.random(100000)
    var ar_3: PythonObject = np.add(ar_1, ar_2).flatten()
    var ar_4: PythonObject = np.multiply(ar_1, ar_2).flatten()


def main():

    # var tens_1:  Tensor[tpe] =  rand[tpe](2)
    # var tens_2:  Tensor[tpe] =  rand[tpe](2)
    # print("Sum")
    # print(tens_1, "\t" ,tens_2, "\n")
    # print(outer_sum(tens_1, tens_2))

    # var tens_3:  Tensor[tpe] =  rand[tpe](2)
    # var tens_4:  Tensor[tpe] =  rand[tpe](2)
    # print("Product")
    # print(tens_3, "\t", tens_4, "\n")
    # print(outer_product(tens_3, tens_4))

    # print("Populating Tensor")
    # var d1_val: Tensor[tpe] = create_tensor(1.0, 2.0)
    # var d2_val: Tensor[tpe] = create_tensor(3.0, 4.0)

    # var d1_prob: Tensor[tpe] = create_tensor(.3, .7)
    # var d2_prob: Tensor[tpe] = create_tensor(.8, .2)



    # var d1_val: Tensor[tpe] = rand[tpe](100)
    # var d2_val: Tensor[tpe] = rand[tpe](100)

    # let d1_prob: Tensor[tpe] = rand[tpe](4)
    # let d2_prob: Tensor[tpe] = rand[tpe](4)

    # var distr1: TensorTuple = TensorTuple(d1_val, d1_prob)
    # var distr2: TensorTuple = TensorTuple(d2_val, d2_prob)
    # var distr3: TensorTuple = convolution(distr1, distr2)


    # 
    # var vec_prob_3: Tensor[tpe] = outer_product_vec(d1_prob, d2_prob)
    # print(d1_prob + d2_prob)
    # print(vec_prob_3)
    # print(distr3.t2)

    # print((d1_prob + d2_prob) == vec_prob_3)

    # var tb1: Tensor[tpe] = rand[tpe](1,2)

    # var tb2: Tensor[tpe] = rand[tpe](2,1)

    # print(tb1)
    # print(tb2)
    # # print(tb1 + tb2)
    # # add[Tensor[tpe]](tb1, tb2)
    py_test()









    return None
