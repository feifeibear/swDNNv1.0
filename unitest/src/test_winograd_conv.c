/***
 * Jerry Fang
 * 2018.9.18
 * Do not forget the shame of our nation
 * **/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include <assert.h>
#include "include/swcommon.h"
#include "include/swwinogradconv.h"

//R = filter_row
//S = filter_col
//P = Co
//Q = Ro
//K = No
//direct_conv with no input pad
void direct_conv(float * D0, float * F, float * O, const int N, const int K, const int P, const int Q, const int C, const int R, const int S) {
    const int P_pad = P + 2; 
    const int Q_pad = Q + 2; 
    int n, k, p, q, c, r, s; 
    float sum; 
    for (n = 0; n < N; n++) {
#pragma omp parallel for
        for (k = 0; k < K; k++) {
            for (p = 1; p < P_pad-1; p++) {
                for (q = 1; q < P_pad-1; q++) {
                    sum = 0; 
#pragma unroll
                    for (c = 0; c < C; c++) {
#pragma unroll
                        for (r = 0; r < R; r++) {
#pragma unroll
                            for (s = 0; s < S; s++) {
                                sum += F[k*C*R*S + c*R*S + r*S + s]*D0[n*C*P_pad*Q_pad + c*P_pad*Q_pad + (p+r-1)*Q_pad + (q+s-1)]; 
                            }
                        }
                    }
                    O[n*K*P*Q+ k*P*Q+ (p-1)*Q+ (q-1)] = sum; 
                }
            }
        }
    }
}


void fjr_direct_conv(float * D0, float * F, float * O, const int N, const int K, const int P, const int Q, const int C, const int R, const int S) {
    const int P_pad = P + 2; 
    const int Q_pad = Q + 2; 
    int n, k, p, q, c, r, s; 
    float sum; 
    for (n = 0; n < N; n++) {
#pragma omp parallel for
        for (k = 0; k < K; k++) {
            for (p = 1; p < P_pad-1; p++) {
                for (q = 1; q < P_pad-1; q++) {
                    sum = 0; 
#pragma unroll
                    for (c = 0; c < C; c++) {
#pragma unroll
                        for (r = 0; r < R; r++) {
#pragma unroll
                            for (s = 0; s < S; s++) {
                                sum += F[r*K*C*S + s*C*K + c*K + k]*\
                                       D0[n*C*P_pad*Q_pad + (p+r-1)*Q_pad*C + (q+s-1)*C + c]; 
                            }
                        }
                    }
                    //O[n*K*P*Q+ k*P*Q+ (p-1)*Q+ (q-1)] = sum; 
                    O[n*K*P*Q+ (p-1)*Q*K + (q-1)*K + k] = sum; 
                }
            }
        }
    }
}


void winograd_conv(const int M, int irows, int C, int K, const int batch, long* total_flops, double* total_time, const int mod, const int verify){
    long i, j, n; 
    const int outHeight = irows-2; 
    const int outWidth = irows-2; 
    const int sizeI = irows*irows; 
    const int sizeF = 3*3; 
    const int sizeO = outHeight*outWidth; 
    const int tiles = (outHeight)*0.5*(outWidth)*0.5; 

    int ret; 

    float* image; 
    float* filter; 
    float* out; 
#ifdef _ALING_MEM_128B_
    image = (float*)_aligned_malloc(batch*C*sizeI*sizeof(float), 128);
    filter = (float*)_aligned_malloc(K*C*sizeF*sizeof(float), 128);
    out = (float*)_aligned_malloc(batch*K*sizeO*sizeof(float), 128);
#else
    image = (float*)malloc(batch*C*sizeI*sizeof(float));
    filter = (float*)malloc(K*C*sizeF*sizeof(float));
    out = (float*)malloc(batch*K*sizeO*sizeof(float));
#endif
    assert(image != NULL); 
    assert(filter != NULL);
    assert(out != NULL);

    //initialize image in parallel
    for(i = 0; i < batch*C*sizeI; i++)
        image[i] = (float)rand()/RAND_MAX; 
    //initialize image in parallel
    for(i = 0; i < K*C*sizeF; i++)
        filter[i] = (float)rand()/RAND_MAX; 

    double timer;
    double timer_acc = 0.0f; 
    struct timeval t1, t2;
    float tt = 0.0; 
    gettimeofday(&t1, NULL);
    sw_winograd_conv(M, image, irows, C, filter, K, batch, out, 0); 
    gettimeofday(&t2, NULL);
    tt = TIME(t1,t2);
    timer_acc += tt;

    timer = timer_acc/1.0f;
    double nflops = (double)batch/1000*K/1000*C/1000*(irows-2)*(irows-2)*3*3*2; 
    double gflops = (double) nflops/timer;
    *total_flops += nflops; 
    *total_time += timer; 

    if(verify){
        printf("Verifying WINOGRAD CONV I = %d Batch = %d C = %d K = %d \n", irows, batch, C, K); 

        float* vout; 
        //allocate data on MCDRAM
        //ret = hbw_posix_memalign((void*)&vout, 64, batch*K*sizeO*sizeof(float)); 
        vout = (float*)malloc(batch*K*sizeO*sizeof(float)); 
        assert(vout != NULL); 
        fjr_direct_conv(image, filter, vout, batch, K, outHeight, outWidth, C, 3, 3); 
        for(n = 0; n < batch*sizeO*K; n++){
            if(fabs(out[n] - vout[n]) > 1e-3){
                printf("Output Error: out[%ld] = %f and vout[%ld] = %f \n", n, out[n], n, vout[n]); 
                break;
            }
        }
        //hbw_free(vout); 
        free(vout); 
    } else
        printf("swBLAS CONV GFLOPS is %.2f \tGFlops \tand timing is \t%f  seconds \n", gflops, timer); 

    timer_acc = 0.0f; 
    gettimeofday(&t1, NULL);
    sw_winograd_conv(M, image, irows, C, filter, K, batch, out, 1);
    gettimeofday(&t2, NULL);
    tt = TIME(t1,t2);
    timer_acc += tt;

    timer = timer_acc/1.0f;
    nflops = (double)batch/1000*K/1000*C/1000*(irows-2)*(irows-2)*3*3*2; 
    gflops = (double) nflops/timer; 
    *total_flops += nflops; 
    *total_time += timer; 

    if(verify){
        printf("Verifying WINOGRAD CONV I = %d Batch = %d C = %d K = %d \n", irows, batch, C, K); 

        float* vout; 
        //allocate data on MCDRAM
        //ret = hbw_posix_memalign((void*)&vout, 64, batch*K*sizeO*sizeof(float)); 
        vout = (float*)malloc(batch*K*sizeO*sizeof(float)); 
        assert(vout != NULL); 
        fjr_direct_conv(image, filter, vout, batch, K, outHeight, outWidth, C, 3, 3); 
        for(n = 0; n < batch*sizeO*K; n++){
            if(fabs(out[n] - vout[n]) > 1e-3){
                printf("Output Error: out[%ld] = %f and vout[%ld] = %f \n", n, out[n], n, vout[n]); 
                break;
            }
        }
        //hbw_free(vout); 
        free(vout); 
    } else
        printf("xMath CONV GFLOPS is %.2f \tGFlops \tand timing is \t%f  seconds \n", gflops, timer); 


#ifdef _ALING_MEM_128B_
    _aligned_free(image);
    _aligned_free(filter);
    _aligned_free(out);
#else
    free(image); 
    free(filter); 
    free(out); 
#endif
}

int profile_winograd(int Ri, int Ni, int No, int batch, int verify){
  int i, j; 
  double timer; 

  int t;
  double total_time = 0.0f;
  long total_flops = 0;

  if (verify) printf("Verifying with Reduced batch size of 8, since direct conv takes long time...\n\n\n\n");

  //falcon_init_lib(32, 512, 512, 258, 258);
  int Ro;
  for(batch = 32; batch <= 128; batch += 96)
    for(Ni = 128; Ni <= 512; Ni += 128)
      for(No = 128; No <= 512; No += 128)
        for(Ro = 32; Ro <= 256; Ro *= 2)
        {
          Ri = Ro + 2;
          printf("B %d Ri %d Ni %d No %d\n", batch, Ri, Ni, No);
          if(verify)
            winograd_conv(1, Ri, Ni, No, 8, &total_flops, &total_time, 50, verify);
          else
            winograd_conv(1, Ri, Ni, No, batch, &total_flops, &total_time, 50, verify);
        }
  //falcon_free_lib();
  return 0;
}


int test_winograd_conv(int Ri, int Ni, int No, int batch, int verify){
  int i, j; 
  double timer; 

  int t;
  double total_time = 0.0f;
  long total_flops = 0;

  if (verify) printf("Verifying with Reduced batch size of 8, since direct conv takes long time...\n\n\n\n");

  //falcon_init_lib(32, 512, 512, 258, 258);
  printf("B %d Ri %d Ni %d No %d\n", batch, Ri, Ni, No);
  if(verify)
    winograd_conv(1, Ri, Ni, No, 8, &total_flops, &total_time, 50, verify);
  else
    winograd_conv(1, Ri, Ni, No, batch, &total_flops, &total_time, 50, verify);
  //falcon_free_lib();
  return 0;
}

