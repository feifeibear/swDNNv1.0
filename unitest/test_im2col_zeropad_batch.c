#include "include/swim2col.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "swblas.h"


void test_im2col_zeropad_batch_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int batch_size) {
  printf("begin test_im2col_zeropad_batch_swblas_float\n");
  int i, j, k;
#define Type float
  struct timeval t1, t2;

  int dilation_h, dilation_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int im2col_batch_size = 1;
  for(i = 1; i <= batch_size; i*=2) {
    if(batch_size%i == 0 && i*((width + 2*pad_w) + output_w)*sizeof(float) < 60*1024)
      im2col_batch_size = i;
  }
  //im2col_batch_size = 1;
  int im_size = channels*height*width;
  int col_size = output_w*output_h*channels*kernel_h*kernel_w;
  int zeropad_col_rowsize = (output_w * output_h + 127)/128*128;
  int zeropad_col_colsize = (kernel_h * kernel_w * channels + 7)/8*8;
  int pad_col_size = zeropad_col_rowsize * zeropad_col_colsize;
  int group_ = 1;

  printf("forward: channels %d, filters %d, height %d, width %d, kernel_h %d, kernel_w %d, \
      pad_h %d, pad_w %d, output_w %d, output_h %d, stride_h %d, stride_w %d, zeropad_col_rowsize %d (%d)\
      zeropad_col_colsize %d (%d), im2col_batch_size %d\n",
          channels, filters, height, width, \
          kernel_h,kernel_w,pad_h,pad_w,output_w,output_h,stride_h,stride_w, zeropad_col_rowsize, output_w*output_h,zeropad_col_colsize,
          kernel_h*kernel_w*channels,
          im2col_batch_size);

  //allocate memory
  Type* data_im = (Type*)malloc(sizeof(Type)*im_size*batch_size);
  for(i = 0; i < im_size*batch_size; ++i )
    data_im[i] = rand()/(Type)RAND_MAX;
#ifdef _MEM_ALIGN_
  float* data_col = (float*)_aligned_malloc(sizeof(Type)*col_size*batch_size, 128);
#elif
  float* data_col = (float*)malloc(sizeof(Type)*col_size*batch_size);
#endif

  if(!data_col)
    printf("allocate data_col failed!\n");
  memset(data_col,0.0, sizeof(Type)*col_size*batch_size);

  Type* zero_pad_data_col = (Type*)malloc(sizeof(Type)*pad_col_size*batch_size);
  if(!zero_pad_data_col)
    printf("allocate zero_pad_data_col failed!\n");
  memset(zero_pad_data_col, 0.0, sizeof(Type)*pad_col_size*batch_size);



  //params for GEMM
  int N = filters;
  int M = zeropad_col_rowsize;
  int K = zeropad_col_colsize;

  int blkK = 0;
  int blkM = 0;
  int blkN = 0;
  int cK, cM, cN;
  for(cK = 8; cK <= K && cK < 512; cK += 8)
    for(cM = 128; cM <= M; cM += 128) {
      for(cN = 32; cN <= N; cN += 32) {
        if(N%cN == 0 && K%cK == 0 && M%cM == 0 && (2*cK*cM + 2*cK*cN + cM*cN)*sizeof(double) < 56*1024*64) {
          blkM = cM;
          blkK = cK;
          blkN = cN;
        }
    }
  }
  printf("im2col M %d K %d N %d blkM %d blkK %d blkN %d\n", M, N, K, blkM, blkK, blkN);

  long output_raw = (long)malloc(sizeof(float)*M*N*batch_size + 128);
  Type* output = (Type*)(output_raw + (128 - (long)output_raw/8%128));
  long weights_raw = (long)malloc(sizeof(float)*N*K + 128);
  Type* weights = (Type*)(weights_raw + (128 - (long)weights_raw/8%128));
  for(i = 0; i < M*N; ++i)
    output[i] = rand()/RAND_MAX;
  for(i = 0; i < N*K; ++i)
    weights[i] = rand()/RAND_MAX;


  //begin im2col
  double im2col_tt = 0;
  gettimeofday(&t1, NULL);
  for(i = 0; i < batch_size; ++i)
    swim2col_f(data_im + i*im_size,channels,height,width,kernel_h,kernel_w,
                  pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_col + col_size*i);
  gettimeofday(&t2, NULL);
  im2col_tt = TIME(t1,t2);
  double total_data_size = (output_w*output_h*kernel_h*kernel_w*channels + channels*height*width)*sizeof(float)*batch_size;
  printf("1.im2col Bandwidth : %lf GB/s, time %lf sec\n", total_data_size/1e9/im2col_tt, im2col_tt);

  double batch_im2col_tt = 0.;
  gettimeofday(&t1, NULL);
  for(i = 0; i < batch_size; i += im2col_batch_size)
    swim2col_zeropad_batch_f(data_im + i*im_size,channels,height,width,kernel_h,kernel_w,
                pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,zero_pad_data_col + i*pad_col_size, im2col_batch_size);
  gettimeofday(&t2, NULL);
  batch_im2col_tt = TIME(t1, t2);
  printf("2.batch im2col Bandwidth : %lf GB/s, time %lf sec\n", total_data_size/1e9/batch_im2col_tt, batch_im2col_tt);

  int cnt = 10;
  double sum1 = 0., sum2 = 0.;
  for(k = 0; k < batch_size; ++k) {
    for(i = 0; i < kernel_h*kernel_w*channels; ++i)
      for(j = 0; j < output_w*output_h; ++j) {
        float a = zero_pad_data_col[k*pad_col_size + i*zeropad_col_rowsize + j];
        float b = data_col[k*col_size + i*output_w*output_h + j];
        if(fabs(a - b) > 1e-3 && cnt) {
          printf("i : %d, j : %d, zeropad %f vs origin %f\n", i, j, a, b);
          cnt--;
        }
        sum1 += a;
        sum2 += b;
      }
  }
  printf("zeropad version pass validation, sum1 %lf, sum2 %lf\n", sum1, sum2);

  gettimeofday(&t1, NULL);
  for(i = 0; i < batch_size; ++i)
    sw_sgemm_trans(zero_pad_data_col + i*pad_col_size, weights, output + i*M*N, M, N, K, blkM, blkN, blkK);
  gettimeofday(&t2, NULL);
  double total_flops = (double)batch_size*(2*(long)M*N*K)/1024/1024/1024;
  double gemm_tt = TIME(t1,t2);
  printf("3.GEMM M %d N %d K %d : %lf Gflops %lf sec\n", M, N, K, total_flops/gemm_tt, gemm_tt);

  gettimeofday(&t1, NULL);
  sw_sgemm_trans(zero_pad_data_col, weights, output, M*batch_size, N, K, blkM, blkN, blkK);
  gettimeofday(&t2, NULL);
  double batch_gemm_tt = TIME(t1,t2);
  printf("4. batch GEMM M %d N %d K %d : %lf Gflops %lf sec\n", M, N, K, total_flops/batch_gemm_tt, batch_gemm_tt);


  double overall_tt = gemm_tt + batch_im2col_tt;
  printf("5.CONV : %lf Gflops %lf sec\n", total_flops/overall_tt, overall_tt);
  printf("============================================================\n");

/*
  gettimeofday(&t1, NULL);
  for(int i = 0; i < 128; ++i)
  caffe::caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M, 
      N, K,
        (float)1., weights, data_col,
        (float)0., output);
  gettimeofday(&t2, NULL);
  total_flops = (double)128*(2*(long)M*N*K)/1024/1024/1024;
  gemm_tt = TIME(t1,t2);
  printf("2.GEMM M %d N %d K %d : %lf Gflops %lf sec\n", M, N, K, total_flops/gemm_tt, gemm_tt);
  overall_tt = gemm_tt + col2im_tt;
  printf("3.BLASCONV : %lf Gflops %lf sec\n", total_flops/overall_tt, overall_tt);
  printf("============================================================\n");
  */
#ifdef _MEM_ALIGN_
  _aligned_free(data_col);
  free(data_im);
  free(zero_pad_data_col);
  free(output_raw);
  free(weights_raw);
#elif
  free(data_col);
  free(data_im);
  free(zero_pad_data_col);
  free(output_raw);
  free(weights_raw);
#endif

#undef Type
}

