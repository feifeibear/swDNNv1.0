#include "include/swtensortrans.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "swblas.h"

void test_tensor_trans_float() {
  int i;
  struct timeval t1, t2;

  int B = 128, H = 32, W = 32, N = 64;
  float* input = _aligned_malloc(sizeof(float)*B*H*W*N, 128);
  float* output = _aligned_malloc(sizeof(float)*B*H*W*N, 128);

  gettimeofday(&t1, NULL);
  image_caffe_to_swdnn_f(input, output, B, N, H, W);
  gettimeofday(&t2, NULL);
  double tt = TIME(t1,t2);
  double total_data_size = B*H*W*N*2*sizeof(float);
  printf("1.Bandwidth : %lf GB/s, time %lf sec\n", total_data_size/1e9/tt, tt);
}
