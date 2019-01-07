extern "C" {
#include "include/swim2col.h"
#include "include/swcommon.h"
#include "./unitest/include/test_im2col.h"
#include "./unitest/include/test_tensortrans.h"
#include "./unitest/include/test_winograd.h"
}
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>


void test_im2col_swblas_main(int B, int Ni, int No, int C, int Pad, int K, int stride) {
  int channels = Ni;
  int filters = No;
  int height = C;
  int width = C;
  int kernel_h = K;
  int kernel_w = K;
  int pad_h = Pad;
  int pad_w = Pad;
  int stride_h = stride;
  int stride_w = stride;
  //sw_blas_init();
  //test_im2col_swblas_float(channels, filters, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
  //test_im2col_zeropad_swblas_float(channels, filters, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, 128);
  test_im2col_zeropad_batch_trans_swblas_float(channels, filters, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, B);
  //test_col2im_swblas_float(channels, filters, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
  //sw_blas_stop();
  return;
}

#define _len_ 7 
int test_im2col_batch() {
  int C[_len_] =      {224, 34, 32, 224, 224, 27, 13};
  int Ni[_len_] =     {3, 64, 64, 3,   3, 96, 256};
  int No[_len_] =     {96, 96, 96, 96,  96, 256, 384};
  int K[_len_] =      {11, 3,  3,  11,  8, 5, 3};
  int Pad[_len_] =    {2, 1,  1,  2,   2, 2, 1};
  int stride[_len_] = {4, 1,  1,  4,   4, 2, 1};
  start_use_slave();
  for(int i = 1; i < _len_; ++i)
    test_im2col_swblas_main(128, Ni[i], No[i], C[i], Pad[i], K[i], stride[i]);
  end_use_slave();
  return 0;
}

/**********
 * A unitest interface, used to compare im2col with implicit-CONV and Winograd
 * In this case, K=3, stride=1, pad=0
 * B in 32 128
 * N in 64, 128, 256, 384, 512
 * C (is input size) in 18 34 66 130 258
 * by Jiarui Fang 2019.1.4
 * *******/
int test_im2col_batch_unitest(int B, int Ni, int No, int C) {
  start_use_slave();
  test_im2col_swblas_main(B, Ni, No, C, 0, 3, 1);
  end_use_slave();
  return 0;
}

int main(int argc, char **argv) {
  std::cout << "begin swDNN unitest" << std::endl;
  std::string case_name = (argv[1]);
  if( case_name == "im2col") {
      if(argc == 6) {
        int B = atoi(argv[2]);
        int Ni = atoi(argv[3]);
        int No  = atoi(argv[4]);
        int C = atoi(argv[5]);
        test_im2col_batch_unitest(B, Ni, No, C);
      } else {
        test_im2col_batch();
      }
  } else if (case_name == "tensortrans") {
    int B = atoi(argv[2]);
    int N = atoi(argv[3]);
    int H  = atoi(argv[4]);
    int W = atoi(argv[5]);
    test_tensor_trans_4D(B, N, H, W);
  } else if (case_name == "winograd") {
    int B = atoi(argv[2]);
    int Ni = atoi(argv[3]);
    int No  = atoi(argv[4]);
    int C = atoi(argv[5]);
    test_winograd_conv(C, Ni, No, B, false); 
  }
  else {
    std::cout << "input a valid case name" << std::endl;
  }
  std::cout << "end swDNN unitest" << std::endl;
  return 0;
}
