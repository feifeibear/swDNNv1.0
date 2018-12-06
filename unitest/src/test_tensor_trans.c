#include "include/swtensortrans.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "swblas.h"

void test_tensor_trans_float() {
  int i, j;
  double tt, total_data_size;
  struct timeval t1, t2;

  int B = 128, H = 32, W = 32, N = 64;
  int buff_size = B*H*W*N;
  float* input = _aligned_malloc(sizeof(float)*B*H*W*N, 128);
  float* output = _aligned_malloc(sizeof(float)*B*H*W*N, 128);

  for(i = 0; i < buff_size; ++i)
    input[i] = rand()/(float)RAND_MAX;
  memset(output, 0, sizeof(float)*buff_size);

  gettimeofday(&t1, NULL);
  image_caffe_to_swdnn_f(input, output, B, N, H, W);
  gettimeofday(&t2, NULL);
  tt = TIME(t1,t2);
  total_data_size = B*H*W*N*4*sizeof(float);
  printf("1.Bandwidth : %lf GB/s, time %lf sec\n", total_data_size/1e9/tt, tt);

  gettimeofday(&t1, NULL);
  swap_lowdim_f(input, output, B, N*H*W);
  gettimeofday(&t2, NULL);
  tt = TIME(t1,t2);
  total_data_size = B*H*W*N*2*sizeof(float);
  printf("1.Bandwidth : %lf GB/s, time %lf sec\n", total_data_size/1e9/tt, tt);

  double sum1 = 0., sum2 = 0.;
  int cnt = 10;
  for(i = 0; i < B; ++i)
    for(j = 0; j < N*H*W; ++j) {
      float a = input[i*N*H*W + j];
      float b = output[j*B + i];
      if(fabs(a - b) > 1e-3 && cnt--)
        printf("a %f vs b %f\n", a, b);
      sum1 += a;
      sum2 += b;
    }
  printf("sum1 %lf, sum2 %lf\n", sum1, sum2);

}
