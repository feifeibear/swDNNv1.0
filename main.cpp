extern "C" {
#include "include/swim2col.h"
#include "include/swcommon.h"
#include "include/swtest.h"
}
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


void test_im2col_swblas_main(int Ni, int No, int C, int Pad, int K, int stride) {
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
  test_im2col_zeropad_batch_trans_swblas_float(channels, filters, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, 128);
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
  for(int i = 0; i < _len_; ++i)
    test_im2col_swblas_main(Ni[i], No[i], C[i], Pad[i], K[i], stride[i]);
  end_use_slave();
  return 0;
}

int main() {
  test_im2col_batch();
  //test_tensor_trans_float();
  return 0;
}
