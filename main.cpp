extern "C" {
#include "include/swim2col.h"
#include "include/swcommon.h"
#include "include/swtest.h"
}
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern "C" {
void test_im2col_zeropad_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w);
}

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
  test_im2col_zeropad_swblas_float(channels, filters, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
  //test_col2im_swblas_float(channels, filters, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);
  //sw_blas_stop();
  return;
}

int main() {
  int Ni = 64;
  int No = 96;
  int C = 224;
  int Pad = 2;
  int K = 11;
  int stride = 1;
  start_use_slave();
  test_im2col_swblas_main(Ni, No, C, Pad, K, stride);
  end_use_slave();
  return 0;
}
