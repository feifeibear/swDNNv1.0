#include "include/swim2col.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


void test_im2col_zeropad_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w) {
  int i, j;
#define Type float
  struct timeval t1, t2;

  int dilation_h, dilation_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int zeropad_col_rowsize = (output_w * output_h + 127)/128*128;
  Type* zero_pad_data_col = (Type*)malloc(sizeof(Type)*zeropad_col_rowsize*channels*kernel_h*kernel_w);
  memset(zero_pad_data_col,0.0, sizeof(Type)*zeropad_col_rowsize*channels*kernel_h*kernel_w);

  Type* data_im = (Type*)malloc(sizeof(Type)*channels*height*width);
  long data_col_raw = (long)malloc(sizeof(Type)*output_w*output_h*channels*kernel_h*kernel_w + 128 );
  Type* data_col = (Type*)(data_col_raw + (128 - (long)data_col_raw/8%128));

  printf("forward: channels %d, filters %d, height %d, width %d, kernel_h %d, kernel_w %d, \
      pad_h %d, pad_w %d, output_w %d, output_h %d, stride_h %d, stride_w %d, zeropad_col_rowsize %d\n",
          channels, filters, height, width, 
          kernel_h,kernel_w,pad_h,pad_w,output_w,output_h,stride_h,stride_w, zeropad_col_rowsize);
  //printf("Now creating image data...\n");
  for(i = 0; i < channels*height*width; ++i )
    data_im[i] = rand()/(Type)RAND_MAX;
//#define PRINT_DATA
#ifdef PRINT_DATA
  printf("Created input data(at channel = 0):\n");
  for( int i = 0; i < height*width; ++i ) {
    if(i%width == 0) printf("\n\t");
    printf("%lf ",data_im[i]);
  }
  printf("\n");
#endif
  //printf("Calling SW im2col...\n");
    //printf("Sum Ref: %lf vs SW: %lf\n",sum_ref,sum);
  //printf("swim2col float test passed.\n");
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
  for(i = 0; i < M*N; ++i)
    output[i] = rand()/RAND_MAX;
  for(i = 0; i < N*K; ++i)
    weights[i] = rand()/RAND_MAX;

  printf("im2col M %d K %d N %d blkM %d blkK %d blkN %d\n", M, N, K, blkM, blkK, blkN);

  double col2im_tt = 0;
  gettimeofday(&t1, NULL);
  swim2col_f(data_im,channels,height,width,kernel_h,kernel_w,
                pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,data_col);
  gettimeofday(&t2, NULL);

  swim2col_zeropad_f(data_im,channels,height,width,kernel_h,kernel_w,
                pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,zero_pad_data_col);

  int cnt = 10;
  double sum1 = 0., sum2 = 0.;
  for(i = 0; i < kernel_h*kernel_w*channels; ++i)
    for(j = 0; j < output_w*output_h; ++j) {
      float a = zero_pad_data_col[i*zeropad_col_rowsize + j];
      float b = data_col[i*output_w*output_h + j];
      if(fabs(a - b) > 1e-3 && cnt) {
        printf("i : %d, j : %d, zeropad %f vs origin %f\n", i, j, a, b);
        cnt--;
      }
      sum1 += a;
      sum2 += b;
    }
  printf("zeropad version pass validation, sum1 %lf, sum2 %lf\n", sum1, sum2);


  double total_data_size = (output_w*output_h*kernel_h*kernel_w*channels + channels*height*width)*sizeof(float);
  col2im_tt = TIME(t1,t2);
  printf("1.im2col Bandwidth : %lf GB/s, time %lf sec\n", total_data_size/1e9/col2im_tt, col2im_tt);

/*
  gettimeofday(&t1, NULL);
  for(int i = 0; i < 128; ++i)
    sw_sgemm_trans(data_col, weights, output, M, N, K, blkM, blkN, blkK);
  gettimeofday(&t2, NULL);
  double total_flops = (double)128*(2*(long)M*N*K)/1024/1024/1024;
  double gemm_tt = TIME(t1,t2);
  printf("2.GEMM M %d N %d K %d : %lf Gflops %lf sec\n", M, N, K, total_flops/gemm_tt, gemm_tt);
  double overall_tt = gemm_tt + col2im_tt;
  printf("3.CONV : %lf Gflops %lf sec\n", total_flops/overall_tt, overall_tt);
  printf("============================================================\n");

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

#undef Type
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
  int stride = 4;
  start_use_slave();
  test_im2col_swblas_main(Ni, No, C, Pad, K, stride);
  end_use_slave();
  return 0;
}
