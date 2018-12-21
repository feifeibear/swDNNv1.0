#include <stdio.h>
// #include <athread.h>
#include <slave.h>
#include <dma.h>
#include <simd.h>
#include <math.h>
#include "../../include/swsoftmax.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

typedef struct TransData_st {
  void* in;
  void* out;
  int tZ;
  int tX;
  int tY;
}TransData;

typedef struct SoftmaxData_st{
  void* bottom_data;
  void* sum_multiplier_;
  void* scale_data;
  void* top_data;
  int channels;
  int dim;
  int outer_num_;
  int inner_num_;
}SoftmaxData;

__thread_local dma_desc dma_get_bottom, dma_get_sumMul, dma_put_top, dma_put_scale;

__thread_local dma_desc dma_get_input, dma_put_output;

#define Type float
// Z x Y x X => Z x X x Y
void swsoftmax_trans_f(TransData *param) {
  // Z/4 x Y/4 x X/4
  int id = athread_get_id(-1);
  int height = id / 16;
  int length =(id % 16)/4;
  int width  =(id % 16)%4;

  int tX  =  param->tX;
  int tY  =  param->tY;
  int tZ  =  param->tZ;

  int tx = tX / 4 + (length<(tX%4));
  int ty = tY / 4 + (width <(tY%4));
  int tz = tZ / 4 + (height<(tZ%4));

  int x_start = length*(tX/4)+(length<(tX%4)?length:(tX%4));
  int y_start = width *(tY/4)+(width <(tY%4)?width :(tY%4));
  int z_start = height*(tY/4)+(height<(tZ%4)?height:(tZ%4));

  Type* in  = (Type*)param->in + z_start*tX*tY + y_start*tX + x_start;
  Type* out = (Type*)param->out+ z_start*tX*tY + x_start*tY + y_start;
if(tx==0||ty==0||tz==0) return ;
  Type* local_input  = (Type*)(long)ldm_malloc(sizeof(Type)*tx*ty);
  Type* local_output = (Type*)(long)ldm_malloc(sizeof(Type)*tx*ty);

  int i,j,k;
  volatile int input_replyget = 0, replyput = 0;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  dma_set_size(&dma_get_input, sizeof(Type)*tx*ty);
  dma_set_bsize(&dma_get_input, sizeof(Type)*tx);
  dma_set_stepsize(&dma_get_input, sizeof(Type)*(tX-tx-1));

  dma_set_size(&dma_put_output, sizeof(Type)*tx*ty);
  dma_set_bsize(&dma_put_output, sizeof(Type)*ty);
  dma_set_stepsize(&dma_put_output, sizeof(Type)*(tY-ty-1));

  for(i=0;i<tz;++i) {
    dma(dma_get_input,(long)(in+i*tX*tY),(long)local_input);
    dma_wait(&input_replyget, 1); input_replyget = 0;

    for(j=0;j<ty;++j) {
      for(k=0;k<tx;++k) {
        local_output[k*ty+j] = local_input[j*tx+k];
      }
    }

    dma(dma_put_output,(long)(out+i*tX*tY),(long)local_output);
    dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_input ,sizeof(Type)*tx*ty);
  ldm_free(local_output,sizeof(Type)*tx*ty);

}

void swsofmax_f(SoftmaxData *param) {
  int id = athread_get_id(-1);
  int channels = param->channels;
  int dim = param->dim;
  int outer_num_ = param->outer_num_;
  int inner_num_ = param->inner_num_;
  int imgSize = channels*inner_num_; // stride
  int local_inner_num = inner_num_/64 + (id<(inner_num_%64));
  int start = id*(inner_num_/64) + (id<(inner_num_%64)?id:(inner_num_%64));

  Type* bottom_data_ptr = (Type*)param->bottom_data + start*channels;
  Type* top_data_ptr = (Type*)param->top_data + start*channels;
  Type* sum_multiplier_ptr = (Type*)param->sum_multiplier_;
  if(local_inner_num==0) return ;
  // IxC
  Type* local_bottom_data = (Type*) (long) ldm_malloc(sizeof(Type)*local_inner_num*channels);
  // IxC
  Type* local_top_data = (Type*) (long) ldm_malloc(sizeof(Type)*local_inner_num*channels);
  // Cx1
  Type* local_sum_multiplier_ = (Type*) (long) ldm_malloc(sizeof(Type)*channels);

  int i,j,k;
  Type stemp;
  Type local_scale;
  // dma set
  volatile int bottom_replyget = 0, sumMul_replyget = 0;
  volatile int top_replyput = 0, scale_replyput = 0;

  dma_set_op(&dma_get_bottom, DMA_GET);
  dma_set_mode(&dma_get_bottom, PE_MODE);
  dma_set_reply(&dma_get_bottom, &bottom_replyget);

  dma_set_op(&dma_get_sumMul, DMA_GET);
  dma_set_mode(&dma_get_sumMul, PE_MODE);
  dma_set_reply(&dma_get_sumMul, &sumMul_replyget);

  dma_set_op(&dma_put_top, DMA_PUT);
  dma_set_mode(&dma_put_top, PE_MODE);
  dma_set_reply(&dma_put_top, &top_replyput);

  dma_set_size(&dma_get_bottom, sizeof(Type)*local_inner_num*channels);
  dma_set_size(&dma_get_sumMul, sizeof(Type)*channels);
  dma_set_size(&dma_put_top, sizeof(Type)*local_inner_num*channels);

  dma(dma_get_sumMul, (long)sum_multiplier_ptr, (long)local_sum_multiplier_ );
  dma_wait(&sumMul_replyget, 1); sumMul_replyget = 0;

  for(i=0;i<outer_num_;++i) {
    dma(dma_get_bottom, (long)(bottom_data_ptr + i*dim),(long)local_bottom_data);
    dma_wait(&bottom_replyget, 1); bottom_replyget = 0;
    for(j=0;j<local_inner_num;++j) {
      local_scale = local_bottom_data[j*channels];
      for(k=1;k<channels;++k) {
        local_scale = max(local_scale,local_bottom_data[j*channels+k]);
      }
      stemp = 0.0;
      for(k=0;k<channels;++k) {
        local_top_data[j*channels+k] = exp(local_bottom_data[j*channels+k] -
                local_sum_multiplier_[k]*local_scale);
        stemp += local_top_data[j*channels+k]*local_sum_multiplier_[k];
      }
      for(k=0;k<channels;++k) {
        local_top_data[j*channels+k] = local_top_data[j*channels+k]/stemp;
      }
    }
    dma(dma_put_top,(long)(top_data_ptr+i*imgSize),(long)local_top_data);
    dma_wait(&top_replyput, 1); top_replyput = 0;
  }

  ldm_free(local_bottom_data,sizeof(Type)*local_inner_num*channels);
  ldm_free(local_top_data, sizeof(Type)*local_inner_num*channels);
  ldm_free(local_sum_multiplier_, sizeof(Type)*channels);
}
/////////////////////////////////
///////////////////////////////////
// old version without transpose
// void softmaxBackward(SlaveSoftmaxParam *pParam)
// {
//   int my_id;
//   volatile unsigned long get_reply,put_reply;
//   int i,j,k,my_col;
//   int outer_num_, channels, inner_num_, dim;
//   float inner_product;
//   float * top_diff;
//   float * top_data;
//   float * bottom_diff;
//   float * scale_data;
// 
//   my_id = athread_get_id(-1);
//   outer_num_ = pParam->outer_num_;
//   channels   = pParam->channels;
//   inner_num_ = pParam->inner_num_;
//   dim        = pParam->dim;
//   top_diff   = pParam->top_diff;
//   top_data   = pParam->top_data;
//   bottom_diff= pParam->bottom_diff;
//   scale_data = pParam->scale_data;
// 
//   float * ldm_top_diff = (float*)(long)ldm_malloc(channels * 4);
//   float * ldm_top_data = (float*)(long)ldm_malloc(channels * 4);
//   float * ldm_bottom_diff = (float*)(long)ldm_malloc(channels * 4);
//   
//   for(k=0; k<outer_num_; ++k)
//   {
//     for(i=0; i<inner_num_; i+=64)
//     {
//       my_col = i+my_id;
//       if(my_col>=inner_num_) break;
//       inner_product=0.0;
//       get_reply = 0;
//       athread_get(PE_MODE, &top_diff[k*dim + my_col],ldm_top_diff,channels*4,&get_reply,0,(inner_num_-1)*sizeof(float),sizeof(float));
//       while(get_reply!=1);
//       get_reply = 0;
//       athread_get(PE_MODE, &top_data[k*dim + my_col],ldm_top_data,channels*4,&get_reply,0,(inner_num_-1)*sizeof(float),sizeof(float));
//       while(get_reply!=1);
//       for(j=0; j<channels; ++j)
//       {
//         inner_product += ldm_top_diff[j] * ldm_top_data[j];
//       }
//       for(j=0; j<channels; ++j)
//       {
//         ldm_bottom_diff[j]=(ldm_top_diff[j]-inner_product)*ldm_top_data[j];
//       }
//       put_reply=0;
//       athread_put(PE_MODE, ldm_bottom_diff, &bottom_diff[k*dim + my_col], channels*4, &put_reply,(inner_num_-1)*sizeof(float),sizeof(float));
//       while(put_reply!=1);
// 
//     }
//   }
// 
//   ldm_free(ldm_top_diff, channels*4);
//   ldm_free(ldm_top_data, channels*4);
//   ldm_free(ldm_bottom_diff, channels * 4);
// }




void softmaxBackward(SlaveSoftmaxParam *pParam)
{
  int my_id;
  volatile unsigned long get_reply,put_reply;
  int i,j,k;
  int outer_num_, channels, inner_num_, dim;
  int buffsize = 16 * 1024;  //16KB
  float inner_product;
  float * top_diff;
  float * top_data;
  float * bottom_diff;
  float * scale_data;
  int numOfGroup, myGroup, myStart, myEnd, groupEachTime, groupSize;

  my_id = athread_get_id(-1);
  outer_num_ = pParam->outer_num_;
  channels   = pParam->channels;
  inner_num_ = pParam->inner_num_;
  dim        = pParam->dim;
  top_diff   = pParam->top_diff;
  top_data   = pParam->top_data;
  bottom_diff= pParam->bottom_diff;
  scale_data = pParam->scale_data;

  numOfGroup = outer_num_ * inner_num_;
  myGroup = numOfGroup>=64?numOfGroup/64:1;
  myStart = my_id * myGroup;
  myEnd = myStart + myGroup;
  groupEachTime = buffsize/(channels * sizeof(float));
  groupSize = groupEachTime;

  // [myStart, myEnd)

  float * ldm_top_diff = (float*)(long)ldm_malloc(buffsize);
  float * ldm_top_data = (float*)(long)ldm_malloc(buffsize);
  float * ldm_bottom_diff = (float*)(long)ldm_malloc(buffsize);

  if(myStart<numOfGroup)
  {
    if(my_id == 63) myEnd = numOfGroup;
    for(i=myStart; i<myEnd; i+= groupEachTime)
    {
      if(i + groupEachTime - 1 >=myEnd) groupSize = myEnd - i;
      get_reply = 0;
      athread_get(PE_MODE, &top_diff[channels * i], ldm_top_diff, groupSize*channels*sizeof(float),&get_reply,0,0,0);
      while(get_reply!=1);
      get_reply = 0;
      athread_get(PE_MODE, &top_data[channels * i], ldm_top_data, groupSize*channels*sizeof(float),&get_reply,0,0,0);
      while(get_reply!=1);
//       for(j=0; j<channels; ++j)
//       {
//         inner_product += ldm_top_diff[j] * ldm_top_data[j];
//       }
//       for(j=0; j<channels; ++j)
//       {
//         ldm_bottom_diff[j]=(ldm_top_diff[j]-inner_product)*ldm_top_data[j];
//       }
      for(j=0;j<groupSize;++j)
      {
        inner_product = 0.0;
        for(k=0;k<channels;++k)
        {
          inner_product += ldm_top_diff[j*channels+k] * ldm_top_data[j*channels+k];
        }
        for(k=0;k<channels;++k)
        {
          ldm_bottom_diff[j*channels+k]=(ldm_top_diff[j*channels+k]-inner_product)*ldm_top_data[j*channels+k];
        }
      }
      put_reply=0;
      athread_put(PE_MODE, ldm_bottom_diff, &bottom_diff[channels * i], groupSize*channels*sizeof(float),&put_reply,0,0);
      while(put_reply!=1);
    }
  }

  ldm_free(ldm_top_diff, buffsize);
  ldm_free(ldm_top_data, buffsize);
  ldm_free(ldm_bottom_diff, buffsize);
}
#undef Type
