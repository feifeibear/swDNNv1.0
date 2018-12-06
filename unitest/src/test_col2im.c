#include "include/swim2col.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void test_col2im_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w) {
#define Type float
  struct timeval t1, t2;
  int i;

  int dilation_h, dilation_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  Type* data_im = (Type*)malloc(sizeof(Type)*channels*height*width);
  long data_col_raw = (long)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w + 128 );
  Type* data_col = (Type*)(data_col_raw + (128*8 - (long)data_col_raw/8%128));
  Type* data_col_ref= (Type*)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w);

  printf("backward: channels %d, filters %d, height %d, width %d, kernel_h %d, kernel_w %d, \
      pad_h %d, pad_w %d, output_w %d, output_h %d, stride_h %d, stride_w %d\n",
          channels, filters, height, width, 
          kernel_h,kernel_w,pad_h,pad_w,output_w,output_h,stride_h,stride_w);
  //printf("Now creating image data...\n");
  for(i = 0; i < channels*height*width; ++i )
    data_im[i] = rand()/(Type)RAND_MAX;

  int N = filters;
  int M = output_w * output_h;
  int K = kernel_h * kernel_w * channels;

  int blkK = 0;
  int blkM = 0;
  int blkN = 0;
  int cK, cM, cN;
  for(cK = 32; cK <= K && cK < 512; cK += 32)
    for(cM = 128; cM <= M; cM += 128) {
      for(cN = 64; cN <= N; cN += 64) {
        if(N%cN == 0 && K%cK == 0 && M%cM == 0 && (2*cK*cM + 2*cK*cN + cM*cN)*sizeof(double) < 56*1024*64) {
          blkM = cM;
          blkK = cK;
          blkN = cN;
        }
    }
  }
  int group_ = 1;
  long output_raw = (long)malloc(sizeof(float)*M*N + 128);
  Type* output = (Type*)(output_raw + (128 - (long)output_raw/8%128));
  long weights_raw = (long)malloc(sizeof(float)*N*K + 128);
  Type* weights = (Type*)(weights_raw + (128 - (long)weights_raw/8%128));

  printf("M %d K %d N %d blkM %d blkK %d blkN %d\n", M, N, K, blkM, blkK, blkN);

  gettimeofday(&t1, NULL);
  /*
  if(blkM != 0)
    for(int i = 0; i < 128; ++i)
      sw_sgemm_trans(data_col, weights, output, M, N, K, blkM, blkN, blkK);
      */
  gettimeofday(&t2, NULL);
  double total_flops = (double)128*(2*(long)M*N*K)/1024/1024/1024;
  double gemm_tt = TIME(t1,t2);
  printf("2.GEMM M %d N %d K %d : %lf Gflops %lf sec\n", M, N, K, total_flops/gemm_tt, gemm_tt);
/*
  gettimeofday(&t1, NULL);
  for(int i = 0; i < 128; ++i)
  caffe::caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M, 
      N, K,
        (float)1., weights, data_col,
        (float)0., output);
  gettimeofday(&t2, NULL);
  total_flops = (double)128*(2*(long)M*N*K)/1024/1024/1024;
  double xmath_tt = TIME(t1,t2);
  printf("2.xMath M %d N %d K %d : %lf Gflops %lf sec\n", M, N, K, total_flops/xmath_tt, xmath_tt);
  */


  gettimeofday(&t1, NULL);
  for(i = 0; i < 128; ++i)
    swcol2im_f(data_col,channels,height,width,kernel_h,kernel_w,
                pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_im);
  gettimeofday(&t2, NULL);
  double total_data_size = 128*(output_w*output_h*kernel_h*kernel_w*channels + channels*height*width)*sizeof(float);
  double col2im_tt = TIME(t1,t2);
  printf("1.col2im_tt Bandwidth : %lf GB/s, time %lf sec\n", total_data_size/1e9/col2im_tt, col2im_tt);

  for(cM = 64; cM <= M; cM += 64) {
    for(cK = 64; cK <= K; cK += 64)
      for(cN = 128; cN <= N; cN += 128) {
        if(N%cN == 0 && K%cK == 0 && M%cM == 0 && (2*cN*cM + 2*cK*cM + cN*cK)*sizeof(double) < 56*1024*64) {
          blkM = cM;
          blkK = cK;
          blkN = cN;
        }
    }
  }
  printf("weight M %d K %d N %d blkM %d blkK %d blkN %d\n", M, K, N, blkM, blkK, blkN);
  /*
  gettimeofday(&t1, NULL);
  //N 128x, M 8x, K 32x
  if(blkN != 0)
  for(int i = 0; i < 128; ++i)
    sw_sgemm_trans(output, data_col, weights, N, K, M, blkN, blkK, blkM);
  gettimeofday(&t2, NULL);
  gemm_tt = TIME(t1,t2);
  printf("weight GEMM M %d N %d K %d : %lf Gflops %lf sec\n", M, N, K, total_flops/gemm_tt, gemm_tt);

 gettimeofday(&t1, NULL);
  //N 128x, M 8x, K 32x
  for(int i = 0; i < 128; ++i)
    caffe::caffe_cpu_gemm<float>(CblasNoTrans, CblasTrans, N, 
      K, M,
        (float)1., output, data_col,
        (float)0., weights);


  gettimeofday(&t2, NULL);
  gemm_tt = TIME(t1,t2);
  printf("weight xMath M %d N %d K %d : %lf Gflops %lf sec\n", M, N, K, total_flops/gemm_tt, gemm_tt);
  */



  free((Type*)output_raw);
  free((Type*)weights_raw);

#undef Type
}
