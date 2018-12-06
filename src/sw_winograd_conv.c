#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>
#include <athread.h>
#include "include/swwinogradconv.h"
#include "../include/swcommon.h"
#include "swblas.h"

//extern SLAVE_FUN(FJR_blas_sgemm)();
//extern SLAVE_FUN(FJR_blas_sgemm_smallB)();
extern int SLAVE_FUN(FJR_input_trans)();
extern int SLAVE_FUN(FJR_input_trans_Ni512)();
extern int SLAVE_FUN(FJR_filter_trans)();
extern int SLAVE_FUN(FJR_output_trans)();

const long MAX_TILES = (MAX_IROWS-2)*(MAX_IROWS-2)*0.25; 
// STRIDE is the max image*C*batch for image
//const long STRIDE = MAX_BATCH*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13); 
//const long STRIDE = ((MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13)); 
#define STRIDE ((MAX_BATCH)*(MAX_IMAGE_CHANNELS+18)*(MAX_TILES+13))
// FSTRIDE is the max C*K for filter
const long FSTRIDE = (MAX_FILTER_CHANNELS+1)*(MAX_FILTERS+1); 

float* t_filter;
float* t_image;
float* c_out;

// setup scratch memory used in the algorithm
void falcon_init_lib(int B, int Ni, int No, int Ci, int Ri){
    int T = (Ci-2)*(Ri-2)/4;
    t_filter = (float*)_aligned_malloc(16*Ni*No*sizeof(float), 128);    
    assert(t_filter != NULL);
    t_image = (float*)_aligned_malloc(16*Ni*T*B*sizeof(float), 128);    
    assert(t_image != NULL);
    c_out = (float*)_aligned_malloc(16*No*T*B*sizeof(float), 128);
    assert(c_out != NULL);
}

// free up the scratch pad
void falcon_free_lib(){
    //free(t_filter);
    //free(t_image);
    //free(c_out);
    _aligned_free(t_filter);
    _aligned_free(t_image);
    _aligned_free(c_out);
}

//input layout (B, Ri, Ci, Ni)
//trans input layout (16, B, T, Ni)
static void fjr_get_tiles(const float* image, float* otile, int N, int C, int ntiles, int Ri, int Ci){
    int t, u;
    int cB, cNi;
    #pragma omp parallel for 
    for(cB = 0; cB < N; ++cB)
      for(cNi = 0; cNi < C; ++cNi) {
        int i, j, x; 
        float tmp[16] __attribute__((aligned(64))); 
        float s[16] __attribute__((aligned(64))); 

        // work on one image plane at a time, irrespective of the order
        int stride = N*C*ntiles;
        int tile_count = cNi + cB*ntiles*C;
        int sizeI = Ri*Ci;
        for(i = 0; i < Ri-2; i += 2){
            #pragma unroll(4)
            for(j = 0; j < Ci-2; j += 2){
                //tmp[0 :4] =data[(i+0)*ldi+j:4]; 
                //tmp[4 :4] =data[(i+1)*ldi+j:4]; 
                //tmp[8 :4] =data[(i+2)*ldi+j:4]; 
                //tmp[12:4] =data[(i+3)*ldi+j:4]; 
                tmp[0] = image[cB*sizeI*C + i*Ci*C + j*C + cNi]; 
                tmp[1] = image[cB*sizeI*C + i*Ci*C + (j+1)*C + cNi]; 
                tmp[2] = image[cB*sizeI*C + i*Ci*C + (j+2)*C + cNi]; 
                tmp[3] = image[cB*sizeI*C + i*Ci*C + (j+3)*C + cNi]; 
                tmp[4] = image[cB*sizeI*C + (i+1)*Ci*C + j*C + cNi]; 
                tmp[5] = image[cB*sizeI*C + (i+1)*Ci*C + (j+1)*C + cNi]; 
                tmp[6] = image[cB*sizeI*C + (i+1)*Ci*C + (j+2)*C + cNi]; 
                tmp[7] = image[cB*sizeI*C + (i+1)*Ci*C + (j+3)*C + cNi]; 
                tmp[8] = image[cB*sizeI*C + (i+2)*Ci*C + j*C + cNi]; 
                tmp[9] = image[cB*sizeI*C + (i+2)*Ci*C + (j+1)*C + cNi]; 
                tmp[10] = image[cB*sizeI*C + (i+2)*Ci*C + (j+2)*C + cNi]; 
                tmp[11] = image[cB*sizeI*C + (i+2)*Ci*C + (j+3)*C + cNi]; 
                tmp[12] = image[cB*sizeI*C + (i+3)*Ci*C + j*C + cNi]; 
                tmp[13] = image[cB*sizeI*C + (i+3)*Ci*C + (j+1)*C + cNi]; 
                tmp[14] = image[cB*sizeI*C + (i+3)*Ci*C + (j+2)*C + cNi]; 
                tmp[15] = image[cB*sizeI*C + (i+3)*Ci*C + (j+3)*C + cNi]; 

                // The tranformation manually simplified
                s[0 ] =(tmp[0] - tmp[8 ]) - (tmp[2 ]- tmp[10]);   
                s[1 ] =(tmp[1] - tmp[9 ]) + (tmp[2 ]- tmp[10]); 
                s[2 ] =(tmp[2] - tmp[10]) - (tmp[1 ]- tmp[9 ]); 
                s[3 ] =(tmp[1] - tmp[9 ]) - (tmp[3 ]- tmp[11]); 
                s[4 ] =(tmp[4] + tmp[8 ]) - (tmp[6 ]+ tmp[10]); 
                s[5 ] =(tmp[5] + tmp[9 ]) + (tmp[6 ]+ tmp[10]); 
                s[6 ] =(tmp[6] + tmp[10]) - (tmp[5 ]+ tmp[9 ]); 
                s[7 ] =(tmp[5] + tmp[9 ]) - (tmp[7 ]+ tmp[11]); 
                s[8 ] =(tmp[8] - tmp[4 ]) - (tmp[10]- tmp[6 ]); 
                s[9 ] =(tmp[9] - tmp[5 ]) + (tmp[10]- tmp[6 ]); 
                s[10] =(tmp[10]- tmp[6 ]) - (tmp[9 ]- tmp[5 ]); 
                s[11] =(tmp[9] - tmp[5 ]) - (tmp[11]- tmp[7 ]); 
                s[12] =(tmp[4] - tmp[12]) - (tmp[6 ]- tmp[14]); 
                s[13] =(tmp[5] - tmp[13]) + (tmp[6 ]- tmp[14]); 
                s[14] =(tmp[6] - tmp[14]) - (tmp[5 ]- tmp[13]); 
                s[15] =(tmp[5] - tmp[13]) - (tmp[7 ]- tmp[15]); 

                // manually unrolled scatter to get max performance
                otile[tile_count+0*stride] = s[0 ]; 
                otile[tile_count+1*stride] = s[1 ]; 
                otile[tile_count+2*stride] = s[2 ]; 
                otile[tile_count+3*stride] = s[3 ]; 
                otile[tile_count+4*stride] = s[4 ]; 
                otile[tile_count+5*stride] = s[5 ]; 
                otile[tile_count+6*stride] = s[6 ]; 
                otile[tile_count+7*stride] = s[7 ]; 
                otile[tile_count+8*stride] = s[8 ]; 
                otile[tile_count+9*stride] = s[9 ]; 
                otile[tile_count+10*stride] = s[10]; 
                otile[tile_count+11*stride] = s[11]; 
                otile[tile_count+12*stride] = s[12]; 
                otile[tile_count+13*stride] = s[13]; 
                otile[tile_count+14*stride] = s[14]; 
                otile[tile_count+15*stride] = s[15]; 

                tile_count += C;
            }
        }
    }
}

// INTERNAL FUNCTION : FORM MATRIX A from input data, also includes transformation
// ldi == irows
// irows = irows
// sizeI = irows * icols
// C == input channel
// otile == results
// N == batch size
// ntiles == #tiles
// image (B, N, H, W) ?
static void get_tiles(const float* image, const int ldi, const int irows, const int sizeI, const int C, float* otile, const int N, const int ntiles){
    int t, u;
    #pragma omp parallel for 
    for(t = 0; t < N*C; t++){
        int i, j, x; 
        float tmp[16] __attribute__((aligned(64))); 
        float s[16] __attribute__((aligned(64))); 

        const float* data = image+t*sizeI; 
        int tile_count = t*ntiles; 
        // work on one image plane at a time, irrespective of the order
        for(i = 0; i < irows-2; i += 2){
            #pragma unroll(4)            
            for(j = 0; j < (irows-2); j += 2){
                //tmp[0 :4] =data[(i+0)*ldi+j:4]; 
                //tmp[4 :4] =data[(i+1)*ldi+j:4]; 
                //tmp[8 :4] =data[(i+2)*ldi+j:4]; 
                //tmp[12:4] =data[(i+3)*ldi+j:4]; 
                tmp[0] =data[(i+0)*ldi+j];
                tmp[1] =data[(i+0)*ldi+j+1];
                tmp[2] =data[(i+0)*ldi+j+2];
                tmp[3] =data[(i+0)*ldi+j+3];

                tmp[4] =data[(i+1)*ldi+j];
                tmp[5] =data[(i+1)*ldi+j+1];
                tmp[6] =data[(i+1)*ldi+j+2];
                tmp[7] =data[(i+1)*ldi+j+3];

                tmp[8] =data[(i+2)*ldi+j];
                tmp[9] =data[(i+2)*ldi+j+1];
                tmp[10] =data[(i+2)*ldi+j+2];
                tmp[11] =data[(i+2)*ldi+j+3];

                tmp[12] =data[(i+3)*ldi+j];
                tmp[13] =data[(i+3)*ldi+j+1];
                tmp[14] =data[(i+3)*ldi+j+2];
                tmp[15] =data[(i+3)*ldi+j+3];

                // The tranformation manually simplified
                s[0 ] =(tmp[0] - tmp[8 ]) - (tmp[2 ]- tmp[10]);
                s[1 ] =(tmp[1] - tmp[9 ]) + (tmp[2 ]- tmp[10]);
                s[2 ] =(tmp[2] - tmp[10]) - (tmp[1 ]- tmp[9 ]);
                s[3 ] =(tmp[1] - tmp[9 ]) - (tmp[3 ]- tmp[11]);
                s[4 ] =(tmp[4] + tmp[8 ]) - (tmp[6 ]+ tmp[10]);
                s[5 ] =(tmp[5] + tmp[9 ]) + (tmp[6 ]+ tmp[10]);
                s[6 ] =(tmp[6] + tmp[10]) - (tmp[5 ]+ tmp[9 ]);
                s[7 ] =(tmp[5] + tmp[9 ]) - (tmp[7 ]+ tmp[11]);
                s[8 ] =(tmp[8] - tmp[4 ]) - (tmp[10]- tmp[6 ]);
                s[9 ] =(tmp[9] - tmp[5 ]) + (tmp[10]- tmp[6 ]);
                s[10] =(tmp[10]- tmp[6 ]) - (tmp[9 ]- tmp[5 ]);
                s[11] =(tmp[9] - tmp[5 ]) - (tmp[11]- tmp[7 ]);
                s[12] =(tmp[4] - tmp[12]) - (tmp[6 ]- tmp[14]);
                s[13] =(tmp[5] - tmp[13]) + (tmp[6 ]- tmp[14]);
                s[14] =(tmp[6] - tmp[14]) - (tmp[5 ]- tmp[13]);
                s[15] =(tmp[5] - tmp[13]) - (tmp[7 ]- tmp[15]);

                // manually unrolled scatter to get max performance
                otile[tile_count+0*STRIDE ] = s[0 ];
                otile[tile_count+1*STRIDE ] = s[1 ];
                otile[tile_count+2*STRIDE ] = s[2 ];
                otile[tile_count+3*STRIDE ] = s[3 ];
                otile[tile_count+4*STRIDE ] = s[4 ];
                otile[tile_count+5*STRIDE ] = s[5 ];
                otile[tile_count+6*STRIDE ] = s[6 ];
                otile[tile_count+7*STRIDE ] = s[7 ];
                otile[tile_count+8*STRIDE ] = s[8 ];
                otile[tile_count+9*STRIDE ] = s[9 ];
                otile[tile_count+10*STRIDE] = s[10];
                otile[tile_count+11*STRIDE] = s[11];
                otile[tile_count+12*STRIDE] = s[12];
                otile[tile_count+13*STRIDE] = s[13];
                otile[tile_count+14*STRIDE] = s[14];
                otile[tile_count+15*STRIDE] = s[15];


                tile_count++; 
            }
        }
    }
}

// INTERNAL FUNCTION: FORM MATRIX B, also includes filter transform
// filter (3, 3, K, C)
// ofilter (16, C, K)
static void fjr_filter_transform(const float* weight, const int C, const int K, float* oweight){
  //filter trans
  float c1[16];
  float F[9];
  int cNo, cNi;
  for(cNo = 0; cNo < K; ++cNo)
    //DMA
    for(cNi = 0; cNi < C; ++cNi){
      F[0] = weight[0*K*C + cNi*K + cNo];
      F[1] = weight[1*K*C + cNi*K + cNo];
      F[2] = weight[2*K*C + cNi*K + cNo];
      F[3] = weight[3*K*C + cNi*K + cNo];
      F[4] = weight[4*K*C + cNi*K + cNo];
      F[5] = weight[5*K*C + cNi*K + cNo];
      F[6] = weight[6*K*C + cNi*K + cNo];
      F[7] = weight[7*K*C + cNi*K + cNo];
      F[8] = weight[8*K*C + cNi*K + cNo];

      c1[0]  = F[0];
      c1[1]  = (F[0]+F[2]+F[1])*0.5f;
      c1[2]  = (F[0]+F[2]-F[1])*0.5f;
      c1[3]  = F[2];
      c1[4]  = (F[0]+F[6]+F[3])*0.5f;
      c1[5]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])+(F[1]+F[7]+F[4]))*0.25f; 
      c1[6]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])-(F[1]+F[7]+F[4]))*0.25f; 
      c1[7]  = (F[2]+F[8]+F[5])*0.5f;
      c1[8]  = (F[0]+F[6]-F[3])*0.5f; 
      c1[9]  = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])+(F[1]+F[7]-F[4]))*0.25f; 
      c1[10] = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])-(F[1]+F[7]-F[4]))*0.25f; 
      c1[11] = (F[2]+F[8]-F[5])*0.5f; 
      c1[12] = F[6]; 
      c1[13] = (F[6]+F[8]+F[7])*0.5f; 
      c1[14] = (F[6]+F[8]-F[7])*0.5f; 
      c1[15] = F[8]; 

      int stride = C*K;
      int x;
      for(x = 0; x < 16; x++){
        oweight[x*stride + cNi*K+ cNo] = c1[x];
      }
    }
}

// INTERNAL FUNCTION: FORM MATRIX B, also includes filter transform
// filter (K, C, 3, 3)
// ofilter (16, K, C)
static void filter_transform(const float* filter, const int C, const int K, float* out){
    int m, n, x; 
    const float *F; 

    #pragma omp parallel for collapse(2) private(m, n, x, F)
    for(m = 0; m < K; m++){
        for(n = 0; n < C; n++){
            float c1[16] __attribute__((aligned(64))); 
            F = filter+n*3*3 + m*3*3*C; 

            // work on in 3x3 plane at a time
            // The tranformation manually simplified
            c1[0]  = F[0]; 
            c1[1]  = (F[0]+F[2]+F[1])*0.5f; 
            c1[2]  = (F[0]+F[2]-F[1])*0.5f; 
            c1[3]  = F[2]; 
            c1[4]  = (F[0]+F[6]+F[3])*0.5f; 
            c1[5]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])+(F[1]+F[7]+F[4]))*0.25f; 
            c1[6]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])-(F[1]+F[7]+F[4]))*0.25f; 
            c1[7]  = (F[2]+F[8]+F[5])*0.5f; 
            c1[8]  = (F[0]+F[6]-F[3])*0.5f; 
            c1[9]  = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])+(F[1]+F[7]-F[4]))*0.25f; 
            c1[10] = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])-(F[1]+F[7]-F[4]))*0.25f; 
            c1[11] = (F[2]+F[8]-F[5])*0.5f; 
            c1[12] = F[6]; 
            c1[13] = (F[6]+F[8]+F[7])*0.5f; 
            c1[14] = (F[6]+F[8]-F[7])*0.5f; 
            c1[15] = F[8]; 

            // scatter
            #pragma unroll(16)
            for(x = 0; x < 16; x++){
                out[x*FSTRIDE+m*C+n] = c1[x]; 
            }
        }
    }
}

// INTERNAL FUNCTION
// GEMM specific to Ist layer of VGG with (M, N, K) = (12544, 64, 3)
// MKL performs bad
static void gemm_ker(int m, int n, int k, const float* a, const int lda, const float* b, const int ldb, float* c, const int ldc){

    const int BLK = 16; 
    int x, xx, y, z, i; 

    for(z = 0; z < n; z++){
        for(x = 0; x < m; x += BLK){
            float p[BLK] __attribute__((aligned(64))); 
            //p[0:BLK] = 0.0f; 
            for(i = 0; i < BLK; ++i)
              p[i] = 0.0f;
            #pragma unroll(3)
            for(y = 0; y < 3; y++){
                #pragma vector aligned
                for(i = 0; i < BLK; i++){
                    p[i] += a[x+i+y*lda]*b[y+z*ldb]; 
                }
            }
            //c[x+z*ldc:BLK] = p[0:BLK]; 
            for(i = 0; i < BLK; ++i)
              c[x + z*ldc + i] = p[i];
        }
    }

}


// INTERNAL FUNCTION
// C = A*B with beta = 0.0f and alpha = 1.0f
// Number of gemm calls is 16*BATCH 
//static void batched_gemm(const float* image, const int irows, const int icols, const float* filter, const int frows, const int fcols, float* out, const int batch){
static void batched_gemm(const float* image, const float* filter, float* out, int No, int Ni, int B, int T, int use_blas){
    int t, i; 
    const char trans ='n'; 
    const float alpha = 1.0; 
    const float beta =  0.0; 
    int blkM = 0;
    for(i = 32; i < 1024; i += 32) {
      if((Ni*i*2 + Ni*No + No*i*2)*sizeof(double) < 64*56*1024 && B*T%i == 0)
        blkM = i;
    }
    if(blkM == 0)
      return;

    //ConvData* params = (ConvData*)malloc(sizeof(ConvData));
    //params->Ni = Ni;
    //params->No = No;
    //params->B = 128;
    //params->T = T*B/128;
    printf("M %d K %d N %d blkM %d: ", T*B, Ni, No, blkM);
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for(i = 0; i < 16; i++){
        if(1) {
          //const float* im = image+i*irows*batch;
          const float* im = image + i*T*B*Ni;
          const float* fi = filter + i*Ni*No;
          float* ot       = out + i*T*B*No;

          if(!use_blas && Ni == 3) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B*T, No, Ni, alpha, im, Ni, fi, No, beta, ot, No);
          } else if(!use_blas && No%128 == 0 && Ni%32 == 0) {
            sw_sgemm(im, fi, ot, B*T, No, Ni, blkM);
          }
          else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B*T, No, Ni, alpha, im, Ni, fi, No, beta, ot, No);
          }
        }
        else {
          for(t = 0; t < B; t++){
              const float* im = image + i*T*B*Ni + t*T*Ni;
              const float* fi = filter + i*Ni*No;
              float* ot       = out + i*T*B*No + t*T*No;
              cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, No, Ni, alpha, im, Ni, fi, No, beta, ot, No);
          }
        }
    }
    gettimeofday(&t2, NULL);
    float tt = TIME(t1,t2);
    float nflops = (float)B*16*T/1000*Ni/1000*No/1000*2;
    double gflops = (double)nflops/tt;

    if(!use_blas && No%128 == 0 && Ni%32 == 0)
      printf("m: %d, n: %d, k: %d; fjrgemm is %lf GFlops, time is %f\n", T, No, Ni, gflops, tt);
    else
      printf("m: %d, n: %d, k: %d; cblas is %lf GFlops, time is %f\n", T, No, Ni, gflops, tt);

    //free(params);
}


//(B, N, H, W)
static void out_transform(const float* d, const int K, const int ntiles, float* out, const int ldo, const int oH, const  int oW, const int N){
    int t; 
    int sizeO = oH*oW; 
    #pragma omp parallel for 
    for(t = 0; t < N*K; t++){
        float c1[16] __attribute__((aligned(64))); 
        float temp[8] __attribute__((aligned(64))); 
        float c2[4] __attribute__((aligned(64))); 
        int tile_offset = t*ntiles; 
        float* data = out +t*sizeO; 
        int i, j;    
        // work on one output plane at a time, irrespective of the order
        for(i = 0; i < oH; i += 2){
            for(j = 0; j < oW; j += 2){
                // gather the 16 elements form C to form a tile
                c1[0 ] = d[tile_offset+0 *STRIDE]; 
                c1[1 ] = d[tile_offset+1 *STRIDE]; 
                c1[2 ] = d[tile_offset+2 *STRIDE]; 
                c1[3 ] = d[tile_offset+3 *STRIDE]; 
                c1[4 ] = d[tile_offset+4 *STRIDE]; 
                c1[5 ] = d[tile_offset+5 *STRIDE]; 
                c1[6 ] = d[tile_offset+6 *STRIDE]; 
                c1[7 ] = d[tile_offset+7 *STRIDE]; 
                c1[8 ] = d[tile_offset+8 *STRIDE]; 
                c1[9 ] = d[tile_offset+9 *STRIDE]; 
                c1[10] = d[tile_offset+10*STRIDE]; 
                c1[11] = d[tile_offset+11*STRIDE]; 
                c1[12] = d[tile_offset+12*STRIDE]; 
                c1[13] = d[tile_offset+13*STRIDE]; 
                c1[14] = d[tile_offset+14*STRIDE]; 
                c1[15] = d[tile_offset+15*STRIDE]; 

                // The tranformation manually simplified
                temp[0] = c1[0]+c1[1]+ c1[2]; 
                temp[1] = c1[1]-c1[2]- c1[3]; 
                temp[2] = c1[4]+c1[5]+ c1[6]; 
                temp[3] = c1[5]-c1[6]- c1[7]; 
                temp[4] = c1[8]+c1[9]+ c1[10]; 
                temp[5] = c1[9]-c1[10]- c1[11]; 
                temp[6] = c1[12]+c1[13]+ c1[14]; 
                temp[7] = c1[13]-c1[14]- c1[15]; 

                c2[0] = temp[0]+temp[2]+temp[4]; 
                c2[1] = temp[1]+temp[3]+temp[5]; 
                c2[2] = temp[2]-temp[4]-temp[6]; 
                c2[3] = temp[3]-temp[5]-temp[7]; 

                data[i*ldo+j]  =c2[0];     
                data[i*ldo+j+1]  =c2[1]; 
                data[(i+1)*ldo+j] = c2[2]; 
                data[(i+1)*ldo+j+1] = c2[3];     
                tile_offset++; 
            }
        }
    }
}

//(16, B, T, No)
//(B, H, W, No)
//
static void fjr_out_transform(const float* d, float* out, int T, int No, int B, int oW, int oH){
//static void fjr_out_transform(const float* d, const int K, const int ntiles, float* out, const int ldo, const int oH, const  int oW, const int N){
    int t;
    int sizeO = oH*oW;
    #pragma omp parallel for
    int cB, cNo;
    for(cB = 0; cB < B; ++cB)
      for(cNo = 0; cNo < No; ++cNo) {
        float c1[16] __attribute__((aligned(64)));
        float temp[8] __attribute__((aligned(64)));
        float c2[4] __attribute__((aligned(64)));
        int i, j;
        int stride = No*B*T;
        int cT = 0;
        // work on one output plane at a time, irrespective of the order
        for(i = 0; i < oH; i += 2){
            for(j = 0; j < oW; j += 2){
                int tile_offset = cB*T*No + cT*No + cNo;
                // gather the 16 elements form C to form a tile
                c1[0 ] = d[tile_offset+0 *stride];
                c1[1 ] = d[tile_offset+1 *stride];
                c1[2 ] = d[tile_offset+2 *stride];
                c1[3 ] = d[tile_offset+3 *stride];
                c1[4 ] = d[tile_offset+4 *stride];
                c1[5 ] = d[tile_offset+5 *stride];
                c1[6 ] = d[tile_offset+6 *stride];
                c1[7 ] = d[tile_offset+7 *stride];
                c1[8 ] = d[tile_offset+8 *stride];
                c1[9 ] = d[tile_offset+9 *stride];
                c1[10] = d[tile_offset+10*stride];
                c1[11] = d[tile_offset+11*stride];
                c1[12] = d[tile_offset+12*stride];
                c1[13] = d[tile_offset+13*stride];
                c1[14] = d[tile_offset+14*stride];
                c1[15] = d[tile_offset+15*stride];

                // The tranformation manuamplified
                temp[0] = c1[0]+c1[1]+ c1[2]; 
                temp[1] = c1[1]-c1[2]- c1[3]; 
                temp[2] = c1[4]+c1[5]+ c1[6]; 
                temp[3] = c1[5]-c1[6]- c1[7]; 
                temp[4] = c1[8]+c1[9]+ c1[10]; 
                temp[5] = c1[9]-c1[10]- c1[11]; 
                temp[6] = c1[12]+c1[13]+ c1[14]; 
                temp[7] = c1[13]-c1[14]- c1[15]; 

                c2[0] = temp[0]+temp[2]+temp[4]; 
                c2[1] = temp[1]+temp[3]+temp[5]; 
                c2[2] = temp[2]-temp[4]-temp[6]; 
                c2[3] = temp[3]-temp[5]-temp[7]; 

                out[cB*No*oH*oW + i*oW*No + j*No + cNo]          = c2[0];
                out[cB*No*oH*oW + (i)*oW*No + (j+1)*No + cNo]    = c2[1];
                out[cB*No*oH*oW + (i+1)*oW*No + j*No + cNo]      = c2[2];
                out[cB*No*oH*oW + (i+1)*oW*No + (j+1)*No + cNo]  = c2[3];

                //data[i*ldo+j]  =c2[0];
                //data[i*ldo+j+1]  =c2[1];
                //data[(i+1)*ldo+j] = c2[2];
                //data[(i+1)*ldo+j+1] = c2[3];

                cT++;
            }
        }
    }
}



//Routine usage interface
void sw_winograd_conv(const int M, float* image, const int irows, const int C, float* filter, const int K, const int batch, float* out, int use_blas){

  const int outHeight = irows-2;
  const int outWidth = irows-2;
  const int sizeI = irows*irows;
  const int tiles = (outHeight)*0.5*(outWidth)*0.5;

  struct timeval t1, t2;
  float tt;

  gettimeofday(&t1, NULL);
  FilterData* filterParams = (FilterData*)malloc(sizeof(FilterData));
  filterParams->filter = filter;
  filterParams->transFilter = t_filter;
  filterParams->Ni = C;
  filterParams->No = K;
  athread_spawn(FJR_filter_trans, filterParams);
  athread_join();
  free(filterParams);
  gettimeofday(&t2, NULL);
  tt = TIME(t1,t2);
  float MBW = (float)4*(9+16)*K*C*1e-9/tt;
  //printf("filter trans time is %lf s, Measured Bandwith %lf Bps\n", tt, MBW);
  //printf("CPE filter trans OK!\n");

  int cnt = 0;
  float sum1 = 0., sum2 = 0.;
//#define CHECK_FILTER_TRANS
#ifdef CHECK_FILTER_TRANS
  float* t_filter_host = (float*)malloc(16*sizeof(float)*C*K);
  memset(t_filter_host, 0.0, 16*sizeof(float)*C*K);
  fjr_filter_transform(filter, C, K, t_filter_host);
  cnt = 0;
  sum1 = 0., sum2 = 0.;
  for(int i = 0; i < 16*C*K; ++i) {
    if(fabs(t_filter[i] - t_filter_host[i]) > 1e-3 && cnt < 10) {
      printf("error @ %d, slave %f vs host %f\n", i, t_filter[i], t_filter_host[i]);
      cnt++;
    }
    sum1 += t_filter_host[i];
    sum2 += t_filter[i];
  }
  printf("sum1 %f, sum2 %f\n", sum1, sum2);
  free(t_filter_host);
#endif

  //printf("begin input trans, Ni %d, B %d, Ci %d, Ri %d\n", C, batch, irows, irows);
  InputData* inputParams = (InputData*)malloc(sizeof(InputData));
  inputParams->input = image;
  inputParams->transInput = t_image;
  inputParams->Ni = C;
  inputParams->B = batch;
  inputParams->Ci = irows;
  inputParams->Ri = irows;
  //athread_spawn(FJR_input_trans_Ni512, inputParams);
  gettimeofday(&t1, NULL);
  athread_spawn(FJR_input_trans, inputParams);
  athread_join();
  gettimeofday(&t2, NULL);
  free(inputParams);

//#define CHECK_INPUT_TRANS
#ifdef CHECK_INPUT_TRANS
  float* t_image_host = (float*)malloc(16*sizeof(float)*tiles*C*batch);
  memset(t_image_host, 0.0, 16*sizeof(float)*tiles*C*batch);
  fjr_get_tiles(image, t_image_host, batch, C, tiles, irows, irows); 
  cnt = 0;
  sum1 = 0., sum2 = 0.;
  for(int i = 0; i < 16*tiles*C*batch; ++i) {
    if(fabs(t_image[i] - t_image_host[i]) > 1e-3 && cnt < 10) {
      printf("error @ %d, slave %f vs host %f\n", i, t_image_host[i], t_image[i]);
      cnt++;
    }
    sum1 += t_image_host[i];
    sum2 += t_image[i];
  }
  printf("input trans sum1 %f, sum2 %f\n", sum1, sum2);
  free(t_image_host);
#endif
  tt = TIME(t1,t2);
  MBW = (float)2*16*tiles*C*batch*4*1e-9/tt;
  printf("input trans time is %lf s, Measured Bandwith %lf Bps\n", tt, MBW);

  batched_gemm(t_image, t_filter, c_out, K, C, batch/M, tiles, use_blas);

  gettimeofday(&t1, NULL);
  OutputData* outputParams = (OutputData*)malloc(sizeof(OutputData));
  outputParams->output = c_out;
  outputParams->transOutput = out;
  outputParams->No = K;
  outputParams->B = batch;
  outputParams->Co = outWidth;
  outputParams->Ro = outHeight;
  athread_spawn(FJR_output_trans, outputParams);
  athread_join();
  free(outputParams);
  gettimeofday(&t2, NULL);
  tt = TIME(t1,t2);
  MBW = (float)(4+16)*K*tiles*batch*4*1e-9/tt;
  printf("output trans time is %lf s, Measured Bandwith %lf Bps\n", tt, MBW);



//#define CHECK_OUTPUT_TRANS
#ifdef CHECK_OUTPUT_TRANS
  float* out_host = (float*)malloc(sizeof(float)*K*batch*outHeight*outWidth);
  memset(out_host, 0.0, sizeof(float)*K*batch*outHeight*outWidth);
  fjr_out_transform(c_out, out_host, tiles, K, batch, outHeight, outWidth);
  cnt = 0;
  sum1 = 0., sum2 = 0.;
  for(int i = 0; i < K*batch*outHeight*outWidth; ++i) {
    if(fabs(out_host[i] - out[i]) > 1e-3 && cnt < 10) {
      printf("error @ %d, slave %f vs host %f\n", i, out_host[i], out[i]);
      cnt++;
    }
    sum1 += out[i];
    sum2 += out_host[i];
  }
  printf("sum1 %f, sum2 %f\n", sum1, sum2);
  free(out_host);
#endif
}

