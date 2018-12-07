extern "C"{
#include "./include/sw_conv_implicit.h"
}
#include "./unitest/include/conv_layer_impl_v2.hpp"
//#include "./unitest/include/conv_layer_impl.hpp"
#include "athread.h"
#include <math.h>
#include <stdio.h>
#include "athread.h"
#include <sys/time.h>
//#define CHECKRES

/*
 * float pad 
 */

//pad in
void test_forward_pad_float() {
  int Ni, No, B, Co, Ro, Ci, Ri, K, pad;
  Ni = 512;
  No = 512;
  B  = 128;
  K  = 3;
  pad = 1;
  Ci = 8;
  Ri = 8;
  Co = Ci+2*pad-K+1;
  Ro = Ri+2*pad-K+1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  float* in = (float*)malloc(sizeof(float)*in_size);
  float* weight = (float*)malloc(sizeof(float)*weight_size);
  float* out = (float*)malloc(sizeof(float)*out_size);
  float* out_ref_ori = (float*)malloc(sizeof(float)*out_size);

  double* in_d = (double*)malloc(sizeof(double)*in_size);
  double* weight_d = (double*)malloc(sizeof(double)*weight_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(float)RAND_MAX;

  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(float)RAND_MAX;

  for( int i = 0; i < in_size; ++i )
    in_d[i] = in[i];

  for( int i = 0; i < weight_size; ++i )
    weight_d[i] = weight[i];

  for( int i = 0; i < out_size; ++i ) {
    out_ref[i] = 0;
    out[i] = 0;
    out_ref_ori[i] = 0;
  }


  for( int st = 0; st < 1; ++st ){

    printf("running sw version pad conv...\n");
    sw_conv_forward_pad_impl_f(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("sw version pad conv OK\n");

    /*
    sw_conv_forward_pad_impl_d(
        in_d,
        weight_d,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("sw version pad conv double OK\n");
    */

    sw_conv_forward_pad_impl_f_ori(
        in,
        weight,
        out_ref_ori,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("inner loop OK!\n");
  }

  printf("calculating errors...\n");
  float sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 ) {
     printf("ERROR at %d: %f vs %f\n", i, out_ref[i], out[i]);
     printf("*********** pad failed ************\n");
     free(out_ref);
     free(out);
     free(in);
     free(weight);
     return ;
   }
   sum += out[i];
   sum_ref += out_ref[i];
  }
  if( fabs(sum_ref - sum) > 1e-4 ) {
     printf("ERROR at SUM: %f vs %f\n", sum_ref, sum);
     printf("*********** pad failed ************\n");
     free(out_ref);
     free(out);
     free(in);
     free(weight);
     return ;
  }
  printf("sum %f vs sum_ref %f athread forward OK!\n", sum, sum_ref);


  free(out_ref);
  free(out);
  free(in);
  free(weight);

  free(weight_d);
  free(in_d);
}

// double pad in
void test_forward_pad() {
  int Ni, No, B, Co, Ro, Ci, Ri, K, pad;
  Ni = 512;
  No = 512;
  B  = 128;
  K  = 3;
  pad = 1;
  Ci = 8;
  Ri = 8;
  Co = Ci+2*pad-K+1;
  Ro = Ri+2*pad-K+1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out = (double*)malloc(sizeof(double)*out_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;

  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;

  for( int i = 0; i < out_size; ++i ) {
    out_ref[i] = 0;
    out[i] = 0;
  }


  for( int st = 0; st < 1; ++st ){
    sw_conv_forward_pad_impl_d(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("sw version pad conv OK\n");

    conv_forward_pad_impl<double>(
        in,
        weight,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("inner loop OK!\n");
  }
  //if(!athread_halt())
  //  printf("athread halt not OK!\n");

  double sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 )
     printf("%lf vs %lf\n", out_ref[i], out[i]);
   sum += out[i];
   sum_ref += out_ref[i];
  }
  printf("sum %lf vs sum_ref %lf athread forward OK!\n", sum, sum_ref);

  free(out_ref);
  free(out);
  free(in);
  free(weight);

}

// no pad
void test_forward() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  Ni = 128;
  No = 256;
  B  = 128;
  Co = 2;
  Ro = 2;
  K  = 3;
  Ci = Co + K - 1;
  Ri = Ro + K - 1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out = (double*)malloc(sizeof(double)*out_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;

  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;


  for( int st = 0; st < 1; ++st ){
    sw_conv_forward_impl_d(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B);

    conv_forward_impl<double>(
        in,
        weight,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B);
    printf("inner loop OK!\n");
  }
  //if(!athread_halt())
  //  printf("athread halt not OK!\n");

  double sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 )
     printf("%lf vs %lf\n", out_ref[i], out[i]);
   sum += out[i];
   sum_ref += out_ref[i];
  }
  free(out_ref);
  free(out);
  free(in);
  free(weight);
  printf("sum %lf vs sum_ref %lf athread forward OK!\n", sum, sum_ref);

}

// no pad
int test_backward() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  Ni = 128;
  No = 128;
  B  = 128;
  Co = 2;
  Ro = 2;
  K  = 3;
  Ci = Co + K - 1;
  Ri = Ro + K - 1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* in_diff = (double*)malloc(sizeof(double)*in_size);
  double* in_diff_ref = (double*)malloc(sizeof(double)*in_size);
  double* weight_diff = (double*)malloc(sizeof(double)*weight_size);
  double* weight_diff_ref = (double*)malloc(sizeof(double)*weight_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out_diff = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(double)RAND_MAX;

  sw_conv_backward_impl_d(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B);

  conv_backward_impl<double>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-4)
      printf("in_diff %lf vs ref %lf\n", in_diff[i], in_diff_ref[i]);

  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-4)
      printf("weight_diff %lf vs ref %lf\n", weight_diff[i], weight_diff_ref[i]);
  printf("backward test OK!");

  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

// in pad out pad double
int test_backward_pad_float() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  int pad = 1;
  Ni = 128;
  No = 128;
  B  = 128;
  Ci = 32; //112;
  Ri = 32; //112;
  //Ci = 4;
  //Ri = 4;
  K  = 3;
  //for mem alloc
  Co = Ci - K+1 + 2*pad;
  Ro = Ri - K+1 + 2*pad;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  printf("before sw_conv_backward_pad_impl_f\n");
  float* in = (float*)malloc(sizeof(float)*in_size);
  float* in_diff = (float*)malloc(sizeof(float)*in_size);
  float* in_diff_ref = (float*)malloc(sizeof(float)*in_size);
  float* weight_diff = (float*)malloc(sizeof(float)*weight_size);
  float* weight_diff_ref = (float*)malloc(sizeof(float)*weight_size);
  float* weight = (float*)malloc(sizeof(float)*weight_size);
  float* out_diff = (float*)malloc(sizeof(float)*out_size);

  printf("after mem alloc\n");
  printf("in_size %d weight_size %d out_size %d\n", in_size, weight_size, out_size);
  for( int i = 0; i < in_size; ++i) {
    in[i] = rand()/(float)RAND_MAX;
  }
  printf("after mem alloc\n");
  for( int i = 0; i < weight_size; ++i)
    weight[i] = rand()/(float)RAND_MAX;
  printf("after mem alloc\n");
  for( int i = 0; i < out_size; ++i)
    out_diff[i] = rand()/(float)RAND_MAX;

  printf("after init\n");
  struct timeval ts, te;
  gettimeofday(&ts, NULL);
  sw_conv_backward_pad_impl_f(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  gettimeofday(&te, NULL);
  double time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
  printf("sw_conv_backward_pad_impl_f OK, time is %lf\n", time);

#ifdef CHECKRES
  conv_backward_pad_impl<float>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-2)
      printf("in_diff %f vs ref %f\n", in_diff[i], in_diff_ref[i]);
  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-2)
      printf("weight_diff %f vs ref %f\n", weight_diff[i], weight_diff_ref[i]);
  printf("backward test OK!");
#endif

  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

int test_backward_pad() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  int pad = 1;
  Ni = 128;
  No = 128;
  B  = 128;
  //Ci = 112;
  //Ri = 112;
  Ci = 8;
  Ri = 8;
  K  = 3;
  K  = 3;
  //for mem alloc
  Co = Ci - K+1 + 2*pad;
  Ro = Ri - K+1 + 2*pad;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* in_diff = (double*)malloc(sizeof(double)*in_size);
  double* in_diff_ref = (double*)malloc(sizeof(double)*in_size);
  double* weight_diff = (double*)malloc(sizeof(double)*weight_size);
  double* weight_diff_ref = (double*)malloc(sizeof(double)*weight_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out_diff = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(double)RAND_MAX;

  struct timeval ts, te;
  gettimeofday(&ts, NULL);
  sw_conv_backward_pad_impl_d(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  gettimeofday(&te, NULL);
  double time = (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0;
  printf("sw_conv_backward_pad_impl_d OK, time is %lf\n", time);
#ifdef CHECKRES
  conv_backward_pad_impl<double>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-4)
      printf("in_diff %lf vs ref %lf\n", in_diff[i], in_diff_ref[i]);
  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-4)
      printf("weight_diff %lf vs ref %lf\n", weight_diff[i], weight_diff_ref[i]);
#endif
  printf("backward test OK!");
  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

int test_backward_pad_split() {
  int Ni, No, B, Co, Ro, Ci, Ri, K;
  int pad = 1;
  Ni = 128;
  No = 128;
  B  = 128;
  Ci = 4;
  Ri = 4;
  K  = 3;
  //for mem alloc
  Co = Ci - K+1 + 2*pad;
  Ro = Ri - K+1 + 2*pad;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* in_diff = (double*)malloc(sizeof(double)*in_size);
  double* in_diff_ref = (double*)malloc(sizeof(double)*in_size);
  double* weight_diff = (double*)malloc(sizeof(double)*weight_size);
  double* weight_diff_ref = (double*)malloc(sizeof(double)*weight_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out_diff = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;
  for( int i = 0; i < out_size; ++i )
    out_diff[i] = rand()/(double)RAND_MAX;

  sw_conv_backward_pad_weight_diff_impl_d(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
  sw_conv_backward_pad_in_diff_impl_d(
        in,
        out_diff,
        weight,
        in_diff,
        weight_diff,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

#ifdef CHECKRES
  conv_backward_pad_impl<double>(
        in,
        out_diff,
        weight,
        in_diff_ref,
        weight_diff_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);

  for( int i = 0; i < in_size; ++i )
    if(fabs(in_diff[i] - in_diff_ref[i]) > 1e-4)
      printf("in_diff %lf vs ref %lf\n", in_diff[i], in_diff_ref[i]);
  for( int i = 0; i < weight_size; ++i )
    if(fabs(weight_diff[i] - weight_diff_ref[i]) > 1e-4)
      printf("weight_diff %lf vs ref %lf\n", weight_diff[i], weight_diff_ref[i]);
#endif
  printf("backward test OK!");
  free(in);
  free(in_diff);
  free(in_diff_ref);
  free(weight_diff);
  free(weight_diff_ref);
  free(weight);
  free(out_diff);
}

void test_conv_backward_pad_impl_f(int Ci, int Ri, int K, int Ni, int No, int B, int pad) {
#define Type float
  // B,Ni,No=128
  printf("------------- Test conv backward-float with pad -------------\n");
  printf("Parameters: Ci=%d, Ri=%d, K=%d, Ni=%d, No=%d, B=%d, pad=%d\n",Ci,Ri,K,Ni,No,B,pad);
  printf("Set up inputs...");
  int Co = Ci+2*pad-K+1;
  int Ro = Ri+2*pad-K+1;
  // input
  Type *in              = (Type*)malloc(sizeof(Type)*B*Ni*Ci*Ri);
  Type *out_grad        = (Type*)malloc(sizeof(Type)*B*No*Co*Ro);
  Type *weight          = (Type*)malloc(sizeof(Type)*K*K*Ni*No);
  Type *in_ref          = (Type*)malloc(sizeof(Type)*B*Ni*Ci*Ri);
  Type *out_grad_ref    = (Type*)malloc(sizeof(Type)*B*No*Co*Ro);
  Type *weight_ref      = (Type*)malloc(sizeof(Type)*K*K*Ni*No);
  // output
  Type *weight_diff     = (Type*)malloc(sizeof(Type)*K*K*Ni*No);
  Type *in_grad         = (Type*)malloc(sizeof(Type)*B*No*Co*Ro);
  Type *weight_diff_ref = (Type*)malloc(sizeof(Type)*K*K*Ni*No);
  Type *in_grad_ref     = (Type*)malloc(sizeof(Type)*B*No*Co*Ro);
  for(int i=0;i<B*Ni*Ci*Ri;++i) {
    in[i] = rand()/(Type)RAND_MAX;
    in_ref[i] = in[i];
  }
  for(int i=0;i<B*No*Co*Ro;++i) {
    out_grad[i] = rand()/(Type)RAND_MAX;
    out_grad_ref[i] = out_grad[i];
  }
  for(int i=0;i<K*K*Ni*No;++i) {
    weight[i]   = rand()/(Type)RAND_MAX;
    weight_ref[i] = weight[i];
  }
  memset(weight_diff,0,sizeof(Type)*K*K*Ni*No);
  memset(weight_diff_ref,0,sizeof(Type)*K*K*Ni*No);
  memset(in_grad,0,sizeof(Type)*B*No*Co*Ro);
  memset(in_grad_ref,0,sizeof(Type)*B*No*Co*Ro);
  printf("Done\n");
#ifdef PRINT_DATA
  printf("DATA in:(1,0)\n");
  for(int i=0;i<Ci*Ri;++i) {
    if(i%Ci==0) printf("\n\t");
    printf("%lf ",in[inGetIdx(1,0,i/Ci,i%Ci,B,Ni,Ri,Ci)]);
  }
  printf("\n\nDATA out_grad:(0,0)\n");
  for(int i=0;i<Co*Ro;++i) {
    if(i%Co==0) printf("\n\t");
    printf("%lf ",out_grad[outGetIdx(0,0,i/Co,i%Co,B,No,Ro,Co)]);
  }
  printf("\n\nDATA weight: No=0,Ni=0\n");
  for(int i=0;i<K*K;++i) {
    if(i%K==0) printf("\n\t");
    printf("%lf ",weight[weightGetIdx(0,0,i/K,i%K,No,Ni,K)]);
  }
  printf("\nNo=1,Ni=0\n");
  for(int i=0;i<K*K;++i) {
    if(i%K==0) printf("\n\t");
    printf("%lf ",weight[weightGetIdx(1,0,i/K,i%K,No,Ni,K)]);
  }
  printf("\n\n");
#endif
  printf("Calling sw_conv_backward_pad_impl_f...\n");
  sw_conv_backward_pad_impl_f(
  //conv_backward_pad_impl_v2<float>(
      in,
      out_grad,
      weight,
      in_grad,
      weight_diff,
      Ci,
      Ri,
      K,
      Ni,
      No,
      B,
      pad);
#ifdef PRINT_DATA
  printf("SW in_grad value:\n");
  for(int i=0;i<Ci*Ri;++i) {
    if(i%Ci==0) printf("\n\t");
    printf("%lf ",in_grad[i]);
  }
  printf("\n\n");
  printf("SW weight_diff value:\n");
  for(int i=0;i<K*K;++i) {
    if(i%K==0) printf("\n\t");
    printf("%lf ",weight_diff[i]);
  }
  printf("\n\n");
#endif
  printf("sw_conv_backward_pad_impl_f End.\n");
  printf("Calliing MPE conv...\n");
  conv_backward_pad_impl<Type>(
      in_ref,
      out_grad_ref,
      weight_ref,
      in_grad_ref,
      weight_diff_ref,
      Ci,
      Ri,
      K,
      Ni,
      No,
      B,
      pad);
#ifdef PRINT_DATA
  printf("Reference in_grad value:\n");
  for(int i=0;i<Ci*Ri;++i) {
    if(i%Ci==0) printf("\n\t");
    printf("%lf ",in_grad_ref[i]);
  }
  printf("\n\n");
  printf("Reference weight_diff value:\n");
  for(int i=0;i<K*K;++i) {
    if(i%K==0) printf("\n\t");
    printf("%lf ",weight_diff_ref[i]);
  }
  printf("\n\n");
#endif
  printf("Reference MPE End\n");
  printf("Calculating Errors...\n");
  int count = 0;
  int t_count;
  double sum = 0.0, sum_ref = 0.0;

  printf("Checking weight_diff...\n");
  count = 0;
  t_count = K*K*Ni*No;
  for(int i=0;i<K*K*Ni*No;++i) {
    if(fabs(weight_diff_ref[i]-weight_diff[i])>1e-2) {
      ++count;
#ifdef  PRINT_WA
#ifndef PRINT_ALL
      if(count<=100) {
#endif
        printf("\tWRONG at (%d,%d,%d,%d): SW: %lf vs Ref: %lf\n",
            i/(K*Ni*No),i%(K*Ni*No)/(Ni*No),
            i%(Ni*No)/No,i%No, weight_diff[i],weight_diff_ref[i]);
#ifdef EXIT_WHEN_WRONG
        printf("*****************TEST Failed*****************\n");
        free(in);
        free(out_grad);
        free(weight);
        free(in_grad);
        free(in_grad_ref);
        free(weight_diff);
        free(weight_diff_ref);
        return ;
#endif
#ifndef PRINT_ALL
      }
#endif
#endif
    }
    sum     += weight_diff[i];
    sum_ref += weight_diff_ref[i];
  }
  printf("weight_diff is OK. ( WA %d in %d )\n",count,t_count);
  printf("weight_diff SUM: SW: %lf vs Ref: %lf\n",sum,sum_ref);
  if(count || fabs(sum-sum_ref)>1e-1) {
    printf("FAIL: weight_diff check\n");
  } else {
    printf("PASS: weight_diff check\n");
  }
  sum = 0.0; sum_ref = 0.0;
  printf("Checking in_grad...\n");
  printf("\t0: SW:%lf vs Ref:%lf\n",in_grad[0],in_grad_ref[0]);
  t_count = B*Ni*Ci*Ri;
  count = 0;
  for(int i=0;i<B*Ni*Ci*Ri;++i) {
    if(fabs(in_grad_ref[i]-in_grad[i])>1e-2) {
      ++count;
#ifdef  PRINT_WA
#ifndef PRINT_ALL
      if(count<=100) {
#endif
        printf("\tWRONG at (%d,%d,%d,%d): SW: %lf vs Ref: %lf\n",
            i/(Ni*Ci*Ri),i%(Ni*Ci*Ri)/(Ci*Ri),
            i%(Ci*Ri)/Ci,i%Ci, in_grad[i],in_grad_ref[i]);
#ifdef EXIT_WHEN_WRONG
        printf("*****************TEST Failed*****************\n");
        free(in);
        free(out_grad);
        free(weight);
        free(in_grad);
        free(in_grad_ref);
        free(weight_diff);
        free(weight_diff_ref);
        return ;
#endif
#ifndef PRINT_ALL
      }
#endif
#endif
    }
    sum     += in_grad[i];
    sum_ref += in_grad_ref[i];
  }
  printf("in_grad is OK. ( WA %d in %d )\n",count,t_count);
  printf("in_grad SUM: SW: %lf vs Ref: %lf\n",sum,sum_ref);
  if(count || fabs(sum-sum_ref)>1e-4) {
    printf("FAIL: in_grad check\n");
  } else {
    printf("PASS: in_grad check\n");
  }
  printf("float conv backward with pad Check complete.\n\n");
  free(in);
  free(out_grad);
  free(weight);
  free(in_grad);
  free(in_grad_ref);
  free(weight_diff);
  free(weight_diff_ref);
#undef Type
}

int test_conv_implicit() {
//printf("TEST conv pad\n");
  //test_conv_pad_main();
  int Ni = 3; //atoi(argv[1]);
  int No = 64; // atoi(argv[2]);
  int C = 224; //atoi(argv[3]);

  //test_forward_pad();
  //test_forward_pad_float();
  //test_forward_pad_float();

  //test_backward_pad();
  test_backward_pad_float();
  //test_backward_pad_split();
  //for(int i = 0; i < 10; ++i)
  //  test_backward_pad_split();

  //test_backward();
  //test_forward();
  return 0;
}
