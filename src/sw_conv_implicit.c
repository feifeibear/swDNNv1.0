/***
 * by Jerry Fang
 * fang_jiarui@163.com
 * For the benefit of the nation,, life and death
 ***/
#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>
#include "./include/sw_conv_implicit.h"
#include "./include/swtensortrans.h"

extern SLAVE_FUN(conv_valid)();
extern SLAVE_FUN(conv_full)();
extern SLAVE_FUN(conv_pad)();
extern SLAVE_FUN(conv_pad_float)();
extern SLAVE_FUN(conv_pad_float__)();
extern SLAVE_FUN(conv_full_pad)();
//extern SLAVE_FUN(conv_full_pad_float)();
extern SLAVE_FUN(conv_full_pad_float_v2)();

//#ifdef SW_TRANS
//#undef SW_TRANS
//#endif
//#define MPE_TRANS

// high -> low
// B, N, R, C
inline int image_caffe_offset(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((n*C + c)*H + h)*W + w);
}
// R, C, N, B
inline int image_swdnn_offset(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((h*W + w)*C + c)*N + n);
}
// R, C, B, N
inline int image_swdnn_offset_back(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((h*W + w)*N + n)*C + c);
}
// No, Ni, Kr, Kc
inline int weight_caffe_offset(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return (( no*Ni + ni )*K + kr)*K + kc;
}
// Kr, Kc, No, Ni
inline int weight_swdnn_offset(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return ((( kr*K + kc )*No + no) * Ni + ni );
}
// Kr, Kc, Ni, No
inline int weight_swdnn_offset_back(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return ((( kr*K + kc )*Ni + ni) * No + no );
}

//#define weight_swdnn_to_caffe(in,out,B,N,H,W) swapBN_HW(in,out,H,W,B,N)
//#define weight_caffe_to_swdnn(in,out,B,N,H,W) swapBN_HW(in,out,B,N,H,W)
//#define image_caffe_to_swdnn_back(in,out,B,N,H,W)  swapBN_HW(in,out,B,N,H,W)
static int init_flag = 0; 
//-----------------------------------
void sw_conv_forward_pad_impl_f_ori(
        const float* in,
        const float* weight,
        float* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
    printf("forward : before swDNN conv float");
#endif
    int i;
    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;
    float* my_in      = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    float* my_out     = (float*)malloc(sizeof(float)*Ro*Co*No*B);
    float* my_weight  = (float*)malloc(sizeof(float)*K*K*No*Ni);

#ifdef MPE_TRANS

#ifdef DEBUG_VERBOSE_SWDNN
    printf("in_trans before");
#endif
    for(cRi = 0; cRi < Ri; ++cRi)
      for(cCi = 0; cCi < Ci; ++cCi)
        for(cNi = 0; cNi < Ni; ++cNi)
          for(cB = 0; cB < B; ++cB)
            my_in[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
              in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#ifdef DEBUG_VERBOSE_SWDNN
    printf("in_trans OVER");
#endif
#elif SW_TRANS
    image_caffe_to_swdnn_f((float*)in,my_in,B,Ni,Ri,Ci);
#else
#endif


#ifdef MPE_TRANS
    for(cNi = 0; cNi < Ni; ++cNi)
      for(cNo = 0; cNo < No; ++cNo)
        for(cKr = 0; cKr < K; ++cKr)
          for(cKc = 0; cKc < K; ++cKc)
              my_weight[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)] = 
                weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
#ifdef DEBUG_VERBOSE_SWDNN
    printf("weight_trans OVER");
#endif
#elif SW_TRANS
    weight_caffe_to_swdnn_f((float*)weight,my_weight,No,Ni,K,K);
#else
#endif

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    param->input =  my_in;
    param->weight = my_weight;
    param->output = my_out;
	  param->_Ni = Ni;
	  param->_Ri = Ri;
	  param->_Ci = Ci;
	  param->_No = No;
	  param->_K  = K;
	  param->_Ro = Ri+2*pad-K+1;
	  param->_Co = Ci+2*pad-K+1;
	  param->_B  = B;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*60*1024/8 - Ni*B-Ni*No)/(No*B);
	  param->_Costride = Costride;
    assert(Costride > 0);
	  int ldm_consume = 8*(Ni*No + No*B*Costride + Ni*B);
	  assert(ldm_consume < 64*1024*64);

    //float impl
	  athread_spawn(conv_pad_float, param);
    //float2double
	  //athread_spawn(conv_pad_float__, param);
	  //athread_spawn(conv_pad, param);
	  athread_join();

#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
      for(cCo = 0; cCo < Co; ++cCo)
        for(cNo = 0; cNo < No; ++cNo)
          for(cB = 0; cB < B; ++cB)
            out[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] =
              my_out[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
    image_swdnn_to_caffe_f(my_out,out,B,No,Ro,Co);
#else
#endif
    free(my_in);
    free(my_weight);
    free(my_out);
    free(param);
	  //printf("forward pad OK\n");
#ifdef DEBUG_VERBOSE_SWDNN
    printf("forward : end swDNN conv float");
#endif
}
//-----------------------------------
void sw_conv_forward_pad_impl_f(
        const float* in,
        const float* weight,
        float* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
    printf("forward : before swDNN conv float\n");
#endif
    int i;
    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;
    float* my_in      = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    float* my_out     = (float*)malloc(sizeof(float)*Ro*Co*No*B);
    float* my_weight  = (float*)malloc(sizeof(float)*K*K*No*Ni);

#ifdef DEBUG_VERBOSE_SWDNN
    printf("in_trans before\n");
#endif
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
      for(cCi = 0; cCi < Ci; ++cCi)
        for(cNi = 0; cNi < Ni; ++cNi)
          for(cB = 0; cB < B; ++cB)
            my_in[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
              in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
    image_caffe_to_swdnn_f((float*)in,my_in,B,Ni,Ri,Ci);
#else
#endif
#ifdef DEBUG_VERBOSE_SWDNN
    printf("in_trans over\n");
#endif


#ifdef MPE_TRANS
    for(cNi = 0; cNi < Ni; ++cNi)
      for(cNo = 0; cNo < No; ++cNo)
        for(cKr = 0; cKr < K; ++cKr)
          for(cKc = 0; cKc < K; ++cKc)
              my_weight[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)] = 
                weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
#elif SW_TRANS
    weight_caffe_to_swdnn_f((float*)weight,my_weight,No,Ni,K,K);
#else
#endif
#ifdef DEBUG_VERBOSE_SWDNN
    printf("weight_trans over\n");
#endif

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
#ifdef DEBUG_VERBOSE_SWDNN
    printf("calc before\n");
#endif
    param->input =  my_in;
    param->weight = my_weight;
    param->output = my_out;
	  param->_Ni = Ni;
	  param->_Ri = Ri;
	  param->_Ci = Ci;
	  param->_No = No;
	  param->_K  = K;
	  param->_Ro = Ri+2*pad-K+1;
	  param->_Co = Ci+2*pad-K+1;
	  param->_B  = B;
    param->_pad = pad;
#ifdef DEBUG_VERBOSE_SWDNN
    printf("calc before\n");
#endif

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*60*1024/8 - Ni*B-Ni*No)/(No*B);
	  param->_Costride = Costride;
    assert(Costride > 0);
	  int ldm_consume = 8*(Ni*No + No*B*Costride + Ni*B);
	  assert(ldm_consume < 64*1024*64);
#ifdef DEBUG_VERBOSE_SWDNN
    printf("calc before\n");
#endif

#ifdef DEBUG_VERBOSE_3
    struct timeval ts, te;
    gettimeofday(&ts, NULL);
#endif
    //float impl
    athread_spawn(conv_pad_float, param);
    //float2double impl
    /*athread_spawn(conv_pad_float__, param);*/
	  //athread_spawn(conv_pad, param);
	  athread_join();
#ifdef DEBUG_VERBOSE_3
    gettimeofday(&te, NULL);
    double time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
#ifdef DEBUG_VERBOSE_SWDNN
    printf("forward swDNN conv float athread time %lf s\n", time);
#endif
#endif
#ifdef DEBUG_VERBOSE_SWDNN
    printf("calc after\n");
#endif

#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
      for(cCo = 0; cCo < Co; ++cCo)
        for(cNo = 0; cNo < No; ++cNo)
          for(cB = 0; cB < B; ++cB)
            out[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] =
              my_out[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
    image_swdnn_to_caffe_f(my_out,out,B,No,Ro,Co);
#else
#endif
    free(my_in);
    free(my_weight);
    free(my_out);
    free(param);
#ifdef DEBUG_VERBOSE_SWDNN
    printf("forward : end swDNN conv float\n");
#endif
}

void sw_conv_forward_pad_impl_d(
        const double* in, 
        const double* weight, 
        double* out,
        //double* bias,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
    printf("forward : before swDNN conv double");
#endif
    int i;
    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;
    double* my_in   = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    double* my_out  = (double*)malloc(sizeof(double)*Ro*Co*No*B);
    double* my_weight = (double*)malloc(sizeof(double)*K*K*No*Ni);
    //double* my_weight_ref = (double*)malloc(sizeof(double)*K*K*No*Ni);

#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
      for(cCi = 0; cCi < Ci; ++cCi)
        for(cNi = 0; cNi < Ni; ++cNi)
          for(cB = 0; cB < B; ++cB)
            my_in[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
              in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
    image_caffe_to_swdnn_d((double*)in,my_in,B,Ni,Ri,Ci);
#else
#endif


#ifdef MPE_TRANS
    for(cNi = 0; cNi < Ni; ++cNi)
      for(cNo = 0; cNo < No; ++cNo)
        for(cKr = 0; cKr < K; ++cKr)
          for(cKc = 0; cKc < K; ++cKc)
              my_weight[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)] = 
                weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
#elif SW_TRANS
    weight_caffe_to_swdnn_d((double*)weight,my_weight,No,Ni,K,K);
#else
#endif

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    param->input =  my_in;
    param->weight = my_weight;
    param->output = my_out;
	  param->_Ni = Ni;
	  param->_Ri = Ri;
	  param->_Ci = Ci;
	  param->_No = No;
	  param->_K  = K;
	  param->_Ro = Ri+2*pad-K+1;
	  param->_Co = Ci+2*pad-K+1;
	  param->_B  = B;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*60*1024/8 - Ni*B-Ni*No)/(No*B);
	  param->_Costride = Costride;
    assert(Costride > 0);
	  int ldm_consume = 8*(Ni*No + No*B*Costride + Ni*B);
	  //printf("ldm comsumption is %d\n", ldm_consume/64);
	  assert(ldm_consume < 64*1024*64);
    //memset(param->output, (double)0, sizeof(double)*Ni*B*Ci*Ri);
	  //printf("befor init forward OK\n");

	  athread_spawn(conv_pad, param);
	  athread_join();

#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
      for(cCo = 0; cCo < Co; ++cCo)
        for(cNo = 0; cNo < No; ++cNo)
          for(cB = 0; cB < B; ++cB)
            out[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] =
              my_out[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
    image_swdnn_to_caffe_d(my_out,out,B,No,Ro,Co);
#else
#endif
/*
    double sum1 = 0, sum2 = 0;
    for( i = 0; i < Ni*No*K*K; ++i ) {
      if( fabs(my_weight_ref[i] - my_weight[i]) > 1e-4) {
       printf("%lf vs %lf\n", my_weight_ref[i], my_weight[i]);
      }
      sum1 += my_weight_ref[i]; sum2 += my_weight[i];
    }
    printf("check over! sum1 %lf and sum2 %lf\n", sum1, sum2);
    exit(0);
    */

    free(my_in);
    free(my_weight);
    free(my_out);
    free(param);
#ifdef DEBUG_VERBOSE_SWDNN
    printf("forward : end swDNN conv double");
#endif
}

void sw_conv_forward_impl_d(
        const double* in, 
        const double* weight, 
        double* out,
        //double* bias,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{
    int i;
    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri-K+1 , Co = Ci-K+1;
    double* my_in   = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    double* my_out  = (double*)malloc(sizeof(double)*Ro*Co*No*B);
    double* my_weight = (double*)malloc(sizeof(double)*K*K*No*Ni);
    //double* my_weight_ref = (double*)malloc(sizeof(double)*K*K*No*Ni);
#ifdef MPE_TRANS 
    for(cRi = 0; cRi < Ri; ++cRi)
      for(cCi = 0; cCi < Ci; ++cCi)
        for(cNi = 0; cNi < Ni; ++cNi)
          for(cB = 0; cB < B; ++cB)
            my_in[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
              in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
    image_caffe_to_swdnn_d((double*)in,my_in,B,Ni,Ri,Ci);
#else
#endif


#ifdef MPE_TRANS 
    for(cNi = 0; cNi < Ni; ++cNi)
      for(cNo = 0; cNo < No; ++cNo)
        for(cKr = 0; cKr < K; ++cKr)
          for(cKc = 0; cKc < K; ++cKc)
              my_weight[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)] = 
                weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
#elif SW_TRANS
    weight_caffe_to_swdnn_d((double*)weight,my_weight,No,Ni,K,K);
#else
#endif

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    param->input =  my_in;
    param->weight = my_weight;
    param->output = my_out;
	  param->_Ni = Ni;
	  param->_Ri = Ri;
	  param->_Ci = Ci;
	  param->_No = No;
	  param->_K  = K;
	  param->_Ro = Ri-K+1;
	  param->_Co = Ci-K+1;
	  param->_B  = B;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

	  int Costride = (64*60*1024/8 - Ni*B*2-Ni*No*2)/(No*B);
	  param->_Costride = Costride;
    assert(Costride > 0);
	  int ldm_consume = 8*(Ni*No*2 + No*B*(Costride) + Ni*B*2);
	  //printf("ldm comsumption is %d\n", ldm_consume/64);
	  assert(ldm_consume < 64*1024*64);
    //memset(param->output, (double)0, sizeof(double)*Ni*B*Ci*Ri);
	  //printf("befor init forward OK\n");

	  athread_spawn(conv_valid, param);
	  athread_join();

#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
      for(cCo = 0; cCo < Co; ++cCo)
        for(cNo = 0; cNo < No; ++cNo)
          for(cB = 0; cB < B; ++cB)
            out[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] =
              my_out[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
    image_swdnn_to_caffe_d(my_out,out,B,No,Ro,Co);
#else
#endif
    free(my_in);
    free(my_weight);
    free(my_out);
    free(param);
	  //printf("forward OK\n");
}


/***
 * conv in backward propagation
 */

void sw_conv_backward_impl_d(
        const double* in,
        const double* out_grad,
        const double* weight,
        double* in_grad,
        double* weight_diff,
        //double* bias_grad,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri-K+1 , Co = Ci-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    double* my_in_grad = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    double* my_in = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    double* my_out_grad = (double*)malloc(sizeof(double)*Ro*Co*No*B);
    double* my_weight_diff = (double*)malloc(sizeof(double)*Ni*No*K*K);

    //Transformation and rot180: in (B, N, R, C) -> (R, C, N, B)
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  //my_in_grad[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                  my_in[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                    in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_caffe_to_swdnn_back_d((double*)in,my_in,B, Ni, Ri, Ci);
#else
#endif


#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  //my_out_grad[image_swdnn_offset_back(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
	image_caffe_to_swdnn_d((double*)out_grad,my_out_grad,B, No, Ro, Co);
#else
#endif

    //memset(my_weight_diff, 0, sizeof(double)*Ni*No*K*K);

    param->input  = my_in;
    param->weight = my_out_grad;
    param->output = my_weight_diff;
	  param->_Ni = B;
	  param->_Ri = Ri;
	  param->_Ci = Ci;
	  param->_No = No;
	  param->_K  = Ci-K+1;
	  param->_Ro = K;
	  param->_Co = K;
	  param->_B  = Ni;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

	  int Costride = (64*55*1024/8-param->_Ni*param->_B*2-
            param->_Ni*param->_No)/
        (param->_No*param->_B);
	  //printf("Costride is %d\n", Costride);
	  param->_Costride = Costride;
    assert(Costride > 0);

    // weight_diff = conv((in), out_grad, 'valid')
	  athread_spawn(conv_valid, param);
	  athread_join();


#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
              weight_diff[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)]
              = my_weight_diff[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_swdnn_to_caffe_d(my_weight_diff, weight_diff,No, Ni, K, K);
#else
#endif
	  //printf("Backward weight_diff OK\n");

    //in_grad TODO should be loaded to 64 CPEs
    //Transforamation and rot180 for Weight
    double* my_weight   = (double*)malloc(sizeof(double)*No*Ni*K*K);
    //double* my_out_grad = (double*)malloc(sizeof(double)*B*No*Co*Ro);

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
                  my_weight[weight_swdnn_offset_back(cNo, cNi, K-1-cKr, K-1-cKc, No, Ni, K)]
                    = weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_caffe_to_swdnn_back_d((double*)weight,my_weight,No, Ni, K, K);
#else
#endif

    param->input  =   my_out_grad;
    param->weight =   my_weight;
    param->output =   my_in_grad;
	  param->_Ni = No;
	  param->_Ri = Ri-K+1;
	  param->_Ci = Ci-K+1;
	  param->_No = Ni;
	  param->_K  = K;
	  param->_Ro = Ri;
	  param->_Co = Ci;
	  param->_B  = B;

    Costride = (64*55*1024/8-param->_Ni*param->_B*2-param->_Ni*param->_No*2)/
            (param->_No*param->_B);
	  param->_Costride = Costride;
	  //printf("Costride is %d\n", Costride);
    assert(Costride > 0);

    //memset(my_in_grad, 0, sizeof(double)*Ni*B*Ci*Ri);
// in_grad = conv(out_grad, rot180(weight), 'full')
	  athread_spawn(conv_full, param);
	  athread_join();
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  in_grad[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] =
                    //my_in_grad[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
                    my_in_grad[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_swdnn_to_caffe_d(my_in_grad,in_grad,B, Ni, Ri, Ci);
#else
#endif
	  //printf("Backward in_grad calc is OK!\n");

    free(my_in_grad);
    free(my_in);
    free(my_weight);
    free(my_out_grad);
    free(my_weight_diff);
    free(param);
}

void sw_conv_backward_pad_impl_f(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        //float* bias_grad,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("backward : begin sw_conv_backward_pad_impl_f\n");
#endif

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    float* my_in = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    float* my_out_grad = (float*)malloc(sizeof(float)*Ro*Co*No*B);
    float* my_weight_diff = (float*)malloc(sizeof(float)*Ni*No*K*K);

    //Transformation and rot180: in (B, N, R, C) -> (R, C, N, B)
#ifdef MPE_TRANS
#ifdef DEBUG_VERBOSE_SWDNN
    printf("MPE_TRANS\n");
#endif
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  my_in[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                  //my_in[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                    in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
#ifdef DEBUG_VERBOSE_SWDNN
    printf("SW_TRANS\n");
#endif
	  image_caffe_to_swdnn_back_f((float*)in,my_in,B, Ni, Ri, Ci);
#else
#ifdef DEBUG_VERBOSE_SWDNN
    printf("Still Caffe data structure.\n");
#endif
#endif


#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
	  image_caffe_to_swdnn_f((float*)out_grad,my_out_grad,B, No, Ro, Co);
#else
#endif

    //memset(my_weight_diff, 0, sizeof(float)*Ni*No*K*K);

    param->input  = my_in;
    param->weight = my_out_grad;
    param->output = my_weight_diff;
	  param->_Ni  = B;
	  param->_Ri  = Ro;//+2*pad-K+1;
	  param->_Ci  = Co;//+2*pad-K+1;
	  param->_No  = No;
	  param->_K   = Ci+2*pad-K+1;
	  param->_Ro  = K;
	  param->_Co  = K;
	  param->_B   = Ni;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*55*1024/8-param->_Ni*param->_B-
            param->_Ni*param->_No)/
        (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    // weight_diff = conv(pad(in), out_grad, 'valid')
#ifdef DEBUG_VERBOSE_SWDNN
    struct timeval ts, te;
    gettimeofday(&ts, NULL);
#endif
	  athread_spawn(conv_pad_float__, param);
	  athread_join();
#ifdef DEBUG_VERBOSE_SWDNN
    gettimeofday(&te, NULL);
    double time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
    double gflop = 2.0*B*Ni*No*K*K*(Ci+2*pad-K+1)*(Co+2*pad-K+1)/1e9;
    printf("Backward swDNN weight_diff float athread time %lf s, gflops %lf GFLOSP, Costride %d \n",
        time, gflop/time, Costride);
#endif

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
              weight_diff[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)]
              = my_weight_diff[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_swdnn_to_caffe_f(my_weight_diff, weight_diff,No, Ni, K, K);
#else
#endif
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("Backward weight_diff OK\n");
#endif

    free(my_weight_diff);
    free(my_in);

    //Transforamation and rot180 for Weight
    float* my_weight   = (float*)malloc(sizeof(float)*No*Ni*K*K);
    float* my_in_grad = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
                  //my_weight[weight_swdnn_offset_back(cNo, cNi, K-1-cKr, K-1-cKc, No, Ni, K)]
                  //my_weight[weight_swdnn_offset(cNo, cNi, K-1-cKr, K-1-cKc, No, Ni, K)] // WRONG
                  my_weight[weight_swdnn_offset(cNi, cNo, K-1-cKr, K-1-cKc, Ni, No, K)]
                    = weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_caffe_to_swdnn_back_f((float*)weight,my_weight,No, Ni, K, K);
#else
#endif
    param->input  =   my_out_grad;
    param->weight =   my_weight;
    param->output =   my_in_grad;
	  param->_Ni = No;
	  param->_Ri = Ro;
	  param->_Ci = Co;
	  param->_No = Ni;
	  param->_K  = K;
	  param->_Ro = Ri;
	  param->_Co = Ci;
	  param->_B  = B;
	  param->_pad  = pad;

    //fjr1buff
    Costride = (64*55*1024/8-param->_Ni*param->_B-param->_Ni*param->_No)/
            (param->_No*param->_B);
	  param->_Costride = Costride;
	  //printf("Costride is %d\n", Costride);
    assert(Costride > 0);

    //memset(my_in_grad, 0, sizeof(float)*Ni*B*Ci*Ri);
    // pad_inv(in_grad) = conv(out_grad, rot180(weight), 'full')
	  //  athread_spawn(conv_full_pad, param);

#ifdef DEBUG_VERBOSE_SWDNN
    gettimeofday(&ts, NULL);
#endif
	  //athread_spawn(conv_full_pad_float,param);
	  athread_spawn(conv_full_pad_float_v2,param);
    athread_join();
#ifdef DEBUG_VERBOSE_SWDNN
    gettimeofday(&te, NULL);
    time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
    printf("Backward swDNN in_diff float athread time %lf s\n", time);
#endif

#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  in_grad[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] =
                    //my_in_grad[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
                    my_in_grad[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_swdnn_to_caffe_f(my_in_grad,in_grad,B, Ni, Ri, Ci);
#else
#endif

    free(my_in_grad);
    free(my_weight);
    free(my_out_grad);
    free(param);

#ifdef DEBUG_VERBOSE_SWDNN
	  printf("backward : end sw_conv_backward_pad_impl_f\n");
#endif
}


void sw_conv_backward_pad_impl_d(
        const double* in,
        const double* out_grad,
        const double* weight,
        double* in_grad,
        double* weight_diff,
        //double* bias_grad,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("begin Backward Pad Impl\n");
#endif

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    double* my_in = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    double* my_out_grad = (double*)malloc(sizeof(double)*Ro*Co*No*B);
    double* my_weight_diff = (double*)malloc(sizeof(double)*Ni*No*K*K);

    //Transformation and rot180: in (B, N, R, C) -> (R, C, N, B)
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  my_in[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                    in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_caffe_to_swdnn_back_d((double*)in,my_in,B, Ni, Ri, Ci);
#else
#endif


#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
	  image_caffe_to_swdnn_d((double*)out_grad,my_out_grad,B, No, Ro, Co);
#else
#endif

    //memset(my_weight_diff, 0, sizeof(double)*Ni*No*K*K);

    param->input  = my_in;
    param->weight = my_out_grad;
    param->output = my_weight_diff;
	  param->_Ni  = B;
	  param->_Ri  = Ro;//+2*pad-K+1;
	  param->_Ci  = Co;//+2*pad-K+1;
	  param->_No  = No;
	  param->_K   = Ci+2*pad-K+1;
	  param->_Ro  = K;
	  param->_Co  = K;
	  param->_B   = Ni;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*55*1024/8-param->_Ni*param->_B-
            param->_Ni*param->_No)/
        (param->_No*param->_B);
	  //int Costride = (64*60*1024/8 - param->_Ni*param->_B*2-param->_Ni*param->_No*2)/(param->_No*param->_B);
    //printf("Costride is %d\n", Costride);
    //int ldm_consume = 8*(param->_Ni*param->_No*2+param->_No*param->_B*Costride+param->_Ni*param->_B*2);
    //assert(ldm_consume < 64*1024*64);
	  param->_Costride = Costride;
    assert(Costride > 0);


    // weight_diff = conv(pad(in), out_grad, 'valid')
#ifdef DEBUG_VERBOSE_3
    struct timeval ts, te;
    gettimeofday(&ts, NULL);
#endif
	  athread_spawn(conv_pad, param);
	  athread_join();
#ifdef DEBUG_VERBOSE_3
    gettimeofday(&te, NULL);
    double time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
    double gflop = 2.0*B*Ni*No*K*K*(Ci+2*pad-K+1)*(Co+2*pad-K+1)/1e9;
#ifdef DEBUG_VERBOSE_SWDNN
    printf("Backward swDNN weight_diff float athread time %lf s, gflops %lf GFLOSP, Costride %d \n",
        time, gflop/time, Costride);
#endif
#endif

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
              weight_diff[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)]
              = my_weight_diff[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_swdnn_to_caffe_d(my_weight_diff, weight_diff,No, Ni, K, K);
#else
#endif
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("Backward weight_diff OK\n");
#endif

    free(my_weight_diff);
    free(my_in);

    //Transforamation and rot180 for Weight
    double* my_weight   = (double*)malloc(sizeof(double)*No*Ni*K*K);
    double* my_in_grad = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
                  my_weight[weight_swdnn_offset_back(cNo, cNi, K-1-cKr, K-1-cKc, No, Ni, K)]
                    = weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_caffe_to_swdnn_back_d((double*)weight,my_weight,No, Ni, K, K);
#else
#endif

    param->input  =   my_out_grad;
    param->weight =   my_weight;
    param->output =   my_in_grad;
	  param->_Ni = No;
	  param->_Ri = Ro;
	  param->_Ci = Co;
	  param->_No = Ni;
	  param->_K  = K;
	  param->_Ro = Ri;
	  param->_Co = Ci;
	  param->_B  = B;
	  param->_pad  = pad;

    //fjr1buff
    Costride = (64*55*1024/8-param->_Ni*param->_B-param->_Ni*param->_No)/
            (param->_No*param->_B);
	  param->_Costride = Costride;
	  //printf("Costride is %d\n", Costride);
    assert(Costride > 0);

    //memset(my_in_grad, 0, sizeof(double)*Ni*B*Ci*Ri);
    // pad_inv(in_grad) = conv(out_grad, rot180(weight), 'full')
	  //  athread_spawn(conv_full_pad, param);
#ifdef DEBUG_VERBOSE_3
    gettimeofday(&ts, NULL);
#endif
	  athread_spawn(conv_full_pad,param);
    athread_join();
#ifdef DEBUG_VERBOSE_3
    gettimeofday(&te, NULL);
    time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
#ifdef DEBUG_VERBOSE_SWDNN
    printf("Backward swDNN in_diff float athread time %lf s\n", time);
#endif
#endif


#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  in_grad[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] =
                    //my_in_grad[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
                    my_in_grad[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_swdnn_to_caffe_d(my_in_grad,in_grad,B, Ni, Ri, Ci);
#else
#endif
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("Backward in_grad calc is OK!\n");
#endif

    free(my_in_grad);
    free(my_weight);
    free(my_out_grad);
    free(param);
}

/***
 * split 2 conv in backward propagation
 */

void sw_conv_backward_pad_weight_diff_impl_d(
        const double* in,
        const double* out_grad,
        const double* weight,
        double* in_grad,
        double* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("backward : begin swDNN weight_diff double\n");
#endif

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    double* my_in = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);
    double* my_out_grad = (double*)malloc(sizeof(double)*Ro*Co*No*B);
    double* my_weight_diff = (double*)malloc(sizeof(double)*Ni*No*K*K);

    //Transformation and rot180: in (B, N, R, C) -> (R, C, N, B)
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  my_in[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                    in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_caffe_to_swdnn_back_d((double*)in,my_in,B, Ni, Ri, Ci);
#else
#endif


#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
	  image_caffe_to_swdnn_d((double*)out_grad,my_out_grad,B, No, Ro, Co);
#else
#endif

    //memset(my_weight_diff, 0, sizeof(double)*Ni*No*K*K);

    param->input  = my_in;
    param->weight = my_out_grad;
    param->output = my_weight_diff;
	  param->_Ni  = B;
	  param->_Ri  = Ro;//+2*pad-K+1;
	  param->_Ci  = Co;//+2*pad-K+1;
	  param->_No  = No;
	  param->_K   = Ci+2*pad-K+1;
	  param->_Ro  = K;
	  param->_Co  = K;
	  param->_B   = Ni;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*55*1024/8-param->_Ni*param->_B-
            param->_Ni*param->_No)/
        (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    // weight_diff = conv(pad(in), out_grad, 'valid')
	  athread_spawn(conv_pad, param);
	  athread_join();

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
              weight_diff[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)]
              = my_weight_diff[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_swdnn_to_caffe_d(my_weight_diff, weight_diff,No, Ni, K, K);
#else
#endif

    free(my_weight_diff);
    free(my_out_grad);
    free(my_in);
    free(param);

#ifdef DEBUG_VERBOSE_SWDNN
	  printf("backward : end swDNN weight_diff double\n");
#endif
}

void sw_conv_backward_pad_in_diff_impl_d(
        const double* in,
        const double* out_grad,
        const double* weight,
        double* in_grad,
        double* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("Backward : begin swDNN in_diff double\n");
#endif

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    double* my_out_grad = (double*)malloc(sizeof(double)*Ro*Co*No*B);

#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
	  image_caffe_to_swdnn_d((double*)out_grad,my_out_grad,B, No, Ro, Co);
#else
#endif
    //Transforamation and rot180 for Weight
    double* my_weight   = (double*)malloc(sizeof(double)*No*Ni*K*K);
    double* my_in_grad = (double*)malloc(sizeof(double)*Ri*Ci*Ni*B);

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
                  my_weight[weight_swdnn_offset_back(cNo, cNi, K-1-cKr, K-1-cKc, No, Ni, K)]
                    = weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_caffe_to_swdnn_back_d((double*)weight,my_weight,No, Ni, K, K);
#else
#endif

    //Ni, No >= 64 %32 = 0 && B >= 128 && B%128=0
    param->input  =   my_out_grad;
    param->weight =   my_weight;
    param->output =   my_in_grad;
	  param->_Ni = No;
	  param->_Ri = Ro;
	  param->_Ci = Co;
	  param->_No = Ni;
	  param->_K  = K;
	  param->_Ro = Ri;
	  param->_Co = Ci;
	  param->_B  = B;
	  param->_pad  = pad;

    //fjr1buff
    int Costride = (64*55*1024/8-param->_Ni*param->_B-param->_Ni*param->_No)/
            (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    //memset(my_in_grad, 0, sizeof(double)*Ni*B*Ci*Ri);
    // pad_inv(in_grad) = conv(out_grad, rot180(weight), 'full')
	  //  athread_spawn(conv_full_pad, param);
	  athread_spawn(conv_full_pad,param);
    athread_join();
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  in_grad[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] =
                    my_in_grad[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_swdnn_to_caffe_d(my_in_grad,in_grad,B, Ni, Ri, Ci);
#else
#endif

    free(my_in_grad);
    free(my_weight);
    free(my_out_grad);
    free(param);
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("Backward : end swDNN in_diff double\n");
#endif
}

void sw_conv_backward_pad_weight_diff_impl_f(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("backward : begin swDNN weight_diff float\n");
#endif

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    float* my_in = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);
    float* my_out_grad = (float*)malloc(sizeof(float)*Ro*Co*No*B);
    float* my_weight_diff = (float*)malloc(sizeof(float)*Ni*No*K*K);

    //Transformation and rot180: in (B, N, R, C) -> (R, C, N, B)
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  my_in[image_swdnn_offset_back(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] = 
                    in[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_caffe_to_swdnn_back_f((float*)in,my_in,B, Ni, Ri, Ci);
#else
#endif


#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
	  image_caffe_to_swdnn_f((float*)out_grad,my_out_grad,B, No, Ro, Co);
#else
#endif

    //memset(my_weight_diff, 0, sizeof(float)*Ni*No*K*K);

    param->input  = my_in;
    param->weight = my_out_grad;
    param->output = my_weight_diff;
	  param->_Ni  = B;
	  param->_Ri  = Ro;//+2*pad-K+1;
	  param->_Ci  = Co;//+2*pad-K+1;
	  param->_No  = No;
	  param->_K   = Ci+2*pad-K+1;
	  param->_Ro  = K;
	  param->_Co  = K;
	  param->_B   = Ni;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*55*1024/8-param->_Ni*param->_B-
            param->_Ni*param->_No)/
        (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    // weight_diff = conv(pad(in), out_grad, 'valid')
    //float impl
    athread_spawn(conv_pad_float, param);

	  //float2double impl
    //athread_spawn(conv_pad_float__, param);
	  athread_join();

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
              weight_diff[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)]
              = my_weight_diff[weight_swdnn_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_swdnn_to_caffe_f(my_weight_diff, weight_diff,No, Ni, K, K);
#else
#endif

    free(my_weight_diff);
    free(my_out_grad);
    free(my_in);
    free(param);

#ifdef DEBUG_VERBOSE_SWDNN
	  printf("backward : end swDNN weight_diff float\n");
#endif
}

void sw_conv_backward_pad_in_diff_impl_f(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("Backward : begin swDNN in_diff float\n");
#endif

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    //weight_diff
    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    float* my_out_grad = (float*)malloc(sizeof(float)*Ro*Co*No*B);

#ifdef MPE_TRANS
    for(cRo = 0; cRo < Ro; ++cRo)
        for(cCo = 0; cCo < Co; ++cCo)
            for(cNo = 0; cNo < No; ++cNo)
                for(cB = 0; cB < B; ++cB)
                  my_out_grad[image_swdnn_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)] = 
                    out_grad[image_caffe_offset(cB, cNo, cRo, cCo, B, No, Ro, Co)];
#elif SW_TRANS
	  image_caffe_to_swdnn_f((float*)out_grad,my_out_grad,B, No, Ro, Co);
#else
#endif
    //Transforamation and rot180 for Weight
    float* my_weight   = (float*)malloc(sizeof(float)*No*Ni*K*K);
    float* my_in_grad = (float*)malloc(sizeof(float)*Ri*Ci*Ni*B);

#ifdef MPE_TRANS
    for(cKr = 0; cKr < K; ++cKr)
        for(cKc = 0; cKc < K; ++cKc)
            for(cNo = 0; cNo < No; ++cNo)
                for(cNi = 0; cNi < Ni; ++cNi){
                  my_weight[weight_swdnn_offset_back(cNo, cNi, K-1-cKr, K-1-cKc, No, Ni, K)]
                    = weight[weight_caffe_offset(cNo, cNi, cKr, cKc, No, Ni, K)];
                }
#elif SW_TRANS
	  weight_caffe_to_swdnn_back_f((float*)weight,my_weight,No, Ni, K, K);
#else
#endif

    //Ni, No >= 64 %32 = 0 && B >= 128 && B%128=0
    param->input  =   my_out_grad;
    param->weight =   my_weight;
    param->output =   my_in_grad;
	  param->_Ni = No;
	  param->_Ri = Ro;
	  param->_Ci = Co;
	  param->_No = Ni;
	  param->_K  = K;
	  param->_Ro = Ri;
	  param->_Co = Ci;
	  param->_B  = B;
	  param->_pad  = pad;

    //fjr1buff
    int Costride = (64*55*1024/8-param->_Ni*param->_B-param->_Ni*param->_No)/
            (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    //memset(my_in_grad, 0, sizeof(float)*Ni*B*Ci*Ri);
    // pad_inv(in_grad) = conv(out_grad, rot180(weight), 'full')
	  //  athread_spawn(conv_full_pad, param);
	  //athread_spawn(conv_full_pad_float,param);
	  athread_spawn(conv_full_pad_float_v2,param);
    athread_join();
#ifdef MPE_TRANS
    for(cRi = 0; cRi < Ri; ++cRi)
        for(cCi = 0; cCi < Ci; ++cCi)
            for(cNi = 0; cNi < Ni; ++cNi)
                for(cB = 0; cB < B; ++cB)
                  in_grad[image_caffe_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)] =
                    my_in_grad[image_swdnn_offset(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)];
#elif SW_TRANS
	  image_swdnn_to_caffe_f(my_in_grad,in_grad,B, Ni, Ri, Ci);
#else
#endif

    free(my_in_grad);
    free(my_weight);
    free(my_out_grad);
    free(param);
#ifdef DEBUG_VERBOSE_SWDNN
	  printf("Backward : end swDNN in_diff float\n");
#endif
}
