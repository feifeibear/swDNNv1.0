#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "simd.h"
#include "dma.h"

#include "../../include/swlstm.h"


#define LONG_PUTR(var,dest) \
asm volatile ("putr %0,%1\n"::"r"(var),"r"(dest):"memory")

#define LONG_GETR(var) \
asm volatile ("getr %0\n":"=r"(var)::"memory")

#define LONG_PUTC(var,dest) \
asm volatile ("putc %0,%1\n"::"r"(var),"r"(dest):"memory")

#define LONG_GETC(var) \
asm volatile ("getc %0\n":"=r"(var)::"memory")

#define SIMDSIZE 4

#define SIMDTYPEF floatv4

#define BUFFSIZE 29*1024

void sync_array(){
    int256 sync_tmp;
    asm volatile(\
        "ldi    %0, 0xff\n"   \
        "sync   %0\n"   \
        "synr   %0\n"   \
        :   \
        :"r"(sync_tmp):"memory");
}

uint64_t slave_get_rtc()
{
    unsigned long rpcc;
    asm volatile ("rcsr %0, 4":"=&r"(rpcc)::"memory");
    return rpcc;
}

float sigmoid(float x) {
  return 1. / (1. + exp(-x));
}

// BUFFSIZE = 29KB
void lstm_slave_noclip_forward_f(LSTMData * param)
{
  float * ldm_pre_gate = (float *)ldm_malloc(BUFFSIZE);
  float * ldm_temp = (float *)ldm_malloc(BUFFSIZE);

  float * pre_gate = ((float*)param->pre_gate_t);
  float * h_to_gate = ((float*)param->h_to_gate);
  float * gate = ((float*)param->gate_t);
  float * h_t = ((float*)param->h_t);
  float * c_t_1 = ((float*)param->c_t_1);
  float * c_t = ((float*)param->c_t);

  SIMDTYPEF vsrc1,vsrc2;
  SIMDTYPEF vdst;
  volatile int get_reply,put_reply;
  int i,j,k,l,batchId,typeId;
  int my_id = athread_get_id(-1);
  int N = param->N_;
  int H = param->H_;
  int t = param->t;

  int sumOfLine = N * 4;
  int simdLength = H / 4 * 4;
  int myLine = sumOfLine>=64?sumOfLine/64:1;
  int myStart = my_id * myLine;
  
  if(myStart<sumOfLine)
  {
    // get_reply=0;
    // athread_get(PE_MODE,clip,ldm_clip,N*sizeof(float),&get_reply,0,0,0);
    // while(get_reply!=1);

    int myEnd = myStart + myLine;
    if(my_id == 63) myEnd = sumOfLine;
    int lineEachTime = BUFFSIZE / (sizeof(float)*H);
    int lineSize = lineEachTime;
    for(i=myStart; i<myEnd; i+=lineEachTime)
    {
        if(i+lineEachTime-1>=myEnd) lineSize = myEnd - i;
        get_reply=0;
        athread_get(PE_MODE,&pre_gate[i*H],ldm_pre_gate,lineSize*H*sizeof(float),&get_reply,0,0,0);
        while(get_reply!=1);
        get_reply=0;
        athread_get(PE_MODE,&h_to_gate[i*H],ldm_temp,lineSize*H*sizeof(float),&get_reply,0,0,0);
        while(get_reply!=1);
        for(j=0;j<lineSize;++j)
        {
            batchId = (i+j)/4;
            if(t)
            {
                for(k=0;k<simdLength;k+=SIMDSIZE)
                {
                    simd_load(vsrc1,&ldm_pre_gate[j*H+k]);
                    simd_load(vsrc2,&ldm_temp[j*H+k]);
                    vdst=vsrc1+vsrc2;
                    simd_store(vdst,&ldm_pre_gate[j*H+k]);
                }
                for(k=simdLength; k<H; ++k)
                {
                    ldm_pre_gate[j*H+k] += ldm_temp[j*H+k];
                }
            }
        }
        for(j<0;j<lineSize;++j)
        {
            batchId = (i+j)/4;
            typeId = (i+j)%4;
            if(typeId == 0)
            {
                for(k=0;k<H;++k)
                {
                    ldm_temp[j*H+k] = sigmoid(ldm_pre_gate[j*H+k]);
                }
            }
            else if(typeId == 1)
            {
               if(t)
               {
                   for(k=0;k<H;++k)
                   {
                       ldm_temp[j*H+k] = sigmoid(ldm_pre_gate[j*H+k]);
                   }
               }
               else
               {
                  vdst = 0.0;
                  for(k=0;k<simdLength;k+=SIMDSIZE)
                  {
                      simd_store(vdst,&ldm_temp[j*H+k]);
                  }
                  for(k=simdLength; k<H; ++k)
                  {
                      ldm_temp[j*H+k] = 0.0;
                  }
               }
            }
            else if(typeId == 2)
            {
                for(k=0;k<H;++k)
                {
                    ldm_temp[j*H+k] = sigmoid(ldm_pre_gate[j*H+k]);
                }
            }
            else
            {
                for(k=0;k<H;++k)
                {
                    ldm_temp[j*H+k] = tanh(ldm_pre_gate[j*H+k]);
                }
            }
        }
        put_reply=0;
        athread_put(PE_MODE,ldm_temp,&gate[i*H],lineSize*H*sizeof(float),&put_reply,0,0);
        while(put_reply!=1);
     }// for
  }


  
  //remain dot

  // float * ldm_c_t = (float*)ldm_realloc(ldm_pre_gate,BUFFSIZE,8*1024);
  // float * ldm_c_t_1 = (float*)ldm_malloc(8*1024);
  // float * ldm_h_t = (float*)ldm_malloc(8*1024);
  float * ldm_c_t = ldm_pre_gate;
  float * ldm_c_t_1 = ldm_pre_gate + 2*1024;
  float * ldm_h_t = ldm_pre_gate + 4*1024;
  int GroupEachTime = BUFFSIZE / (sizeof(float)*4*H);
  int groupSize = GroupEachTime;

  int myGroup = N>=64?N/64:1;
  myStart = my_id * myGroup * 4;
  if(myStart<N)
  {
      int myEnd = myStart + myGroup * 4;
      if(my_id==63) myEnd = N;
      for(i=myStart; i<myEnd; i+=GroupEachTime*4)
      {
          if(i+GroupEachTime*4-1>=myEnd) groupSize = (myEnd-i)/4;
          get_reply=0;
          athread_get(PE_MODE,&gate[i*H],ldm_temp,sizeof(float)*H*4*groupSize,&get_reply,0,0,0);
          while(get_reply!=1);
          get_reply=0;
          athread_get(PE_MODE,&c_t_1[i/4*H],ldm_c_t_1,sizeof(float)*H*groupSize,&get_reply,0,0,0);
          while(get_reply!=1);
          for(j=0;j<groupSize;++j) //control 4 lines
          {
              //gate_t[d]       : ldm_temp[j*4*H]
              //gate_t[d+H]     : ldm_temp[j*4*H + H]
              //gate_t[d+2*H]   : ldm_temp[j*4*H + 2*H]
              //gate_t[d+3*H]   : ldm_temp[j*4*H + 3*H]
              for(k=0;k<H;++k)
              {
                  ldm_c_t[j*H+k] = ldm_temp[j*4*H + H + k] * ldm_c_t_1[j*H+k] + ldm_temp[j*4*H + k] * ldm_temp[j*4*H + 3*H + k];
                  ldm_h_t[j*H+k] = ldm_temp[j*4*H + 2*H + k] * tanh(ldm_c_t[j*H+k]);
                  //ldm_h_t[j*H+k] = 0.0;
              }
          }
          put_reply=0;
          athread_put(PE_MODE,ldm_c_t,&c_t[i/4*H],sizeof(float)*H*groupSize,&put_reply,0,0);
          while(put_reply!=1);
          put_reply=0;
          athread_put(PE_MODE,ldm_h_t,&h_t[i/4*H],sizeof(float)*H*groupSize,&put_reply,0,0);
          while(put_reply!=1);
      }


  }

  



  ldm_free(ldm_pre_gate,BUFFSIZE);
  ldm_free(ldm_temp,BUFFSIZE);

}


//BUFFSIZE = 29KB
void lstm_slave_clip_forward_f(LSTMData * param)
{
  float * ldm_pre_gate = (float *)ldm_malloc(BUFFSIZE);
  float * ldm_temp = (float *)ldm_malloc(BUFFSIZE);
  float * ldm_clip = (float *)ldm_malloc(2*1024);

  float * clip = ((float*)param->clip_t);
  float * pre_gate = ((float*)param->pre_gate_t);
  float * h_to_gate = ((float*)param->h_to_gate);
  float * gate = ((float*)param->gate_t);
  float * h_t = ((float*)param->h_t);
  float * c_t_1 = ((float*)param->c_t_1);
  float * c_t = ((float*)param->c_t);

  int N = param->N_;
  int H = param->H_;

  SIMDTYPEF vsrc1,vsrc2;
  SIMDTYPEF vdst;
  volatile int get_reply,put_reply;
  int i,j,k,l,batchId,typeId;
  int my_id = athread_get_id(-1);


  int sumOfLine = N * 4;
  int simdLength = H / 4 * 4;
  int myLine = sumOfLine>=64?sumOfLine/64:1;
  int myStart = my_id * myLine;

  
  if(myStart<sumOfLine)
  {

    get_reply=0;
    athread_get(PE_MODE,clip,ldm_clip,N*sizeof(float),&get_reply,0,0,0);
    while(get_reply!=1);

    int myEnd = myStart + myLine;
    if(my_id == 63) myEnd = sumOfLine;
    int lineEachTime = BUFFSIZE / (sizeof(float)*H);
    int lineSize = lineEachTime;
    for(i=myStart; i<myEnd; i+=lineEachTime)
    {
        if(i+lineEachTime-1>=myEnd) lineSize = myEnd - i;
        get_reply=0;
        athread_get(PE_MODE,&pre_gate[i*H],ldm_pre_gate,lineSize*H*sizeof(float),&get_reply,0,0,0);
        while(get_reply!=1);
        get_reply=0;
        athread_get(PE_MODE,&h_to_gate[i*H],ldm_temp,lineSize*H*sizeof(float),&get_reply,0,0,0);
        while(get_reply!=1);
        for(j=0;j<lineSize;++j)
        {
            batchId = (i+j)/4;
            if(ldm_clip[batchId])
            {
                for(k=0;k<simdLength;k+=SIMDSIZE)
                {
                    simd_load(vsrc1,&ldm_pre_gate[j*H+k]);
                    simd_load(vsrc2,&ldm_temp[j*H+k]);
                    vdst=vsrc1+vsrc2;
                    simd_store(vdst,&ldm_pre_gate[j*H+k]);
                }
                for(k=simdLength; k<H; ++k)
                {
                    ldm_pre_gate[j*H+k] += ldm_temp[j*H+k];
                }
            }
        }


        for(j=0;j<lineSize;++j)
        {
            batchId = (i+j)/4;
            typeId = (i+j)%4;
            if(typeId == 0)
            {
                for(k=0;k<H;++k)
                {
//                    ldm_temp[j*H+k] = sigmoid(ldm_pre_gate[j*H+k]);
                }
            }
            else if(typeId == 1)
            {
               if(ldm_clip[batchId])
               {
                   for(k=0;k<H;++k)
                   {
//                       ldm_temp[j*H+k] = sigmoid(ldm_pre_gate[j*H+k]);
                   }

               }
               else
               {
                  vdst = 0.0;
                  for(k=0;k<simdLength;k+=SIMDSIZE)
                  {
                      simd_store(vdst,&ldm_temp[j*H+k]);
                  }
                  for(k=simdLength; k<H; ++k)
                  {
                      ldm_temp[j*H+k] = 0.0;
                  }

               }
            }
            else if(typeId == 2)
            {
                for(k=0;k<H;++k)
                {
  //                  ldm_temp[j*H+k] = sigmoid(ldm_pre_gate[j*H+k]);
                }
            }
            else
            {
                for(k=0;k<H;++k)
                {
                    ldm_temp[j*H+k] = tanh(ldm_pre_gate[j*H+k]);
                }
            }
        }
        put_reply=0;
        athread_put(PE_MODE,ldm_temp,&gate[i*H],lineSize*H*sizeof(float),&put_reply,0,0);
        while(put_reply!=1);
     }// for
  }
  
  //remain dot
  // float * ldm_c_t = (float*)ldm_realloc(ldm_pre_gate,BUFFSIZE,8*1024);
  // float * ldm_c_t_1 = (float*)ldm_malloc(8*1024);
  // float * ldm_h_t = (float*)ldm_malloc(8*1024);
  float * ldm_c_t = ldm_pre_gate;
  float * ldm_c_t_1 = ldm_pre_gate + 2*1024;
  float * ldm_h_t = ldm_pre_gate + 4*1024;

  int GroupEachTime = BUFFSIZE / (sizeof(float)*4*H);
  int groupSize = GroupEachTime;

  int myGroup = N>=64?N/64:1;
  myStart = my_id * myGroup * 4;

  if(myStart<4*N)
  {
      int myEnd = myStart + myGroup * 4;
      if(my_id==63) myEnd = 4*N;
      for(i=myStart; i<myEnd; i+=GroupEachTime*4)
      {
          if(i+GroupEachTime*4-1>=myEnd) groupSize = (myEnd-i)/4;
          get_reply=0;
          athread_get(PE_MODE,&gate[i*H],ldm_temp,sizeof(float)*H*4*groupSize,&get_reply,0,0,0);
          while(get_reply!=1);
          get_reply=0;
          athread_get(PE_MODE,&c_t_1[i/4*H],ldm_c_t_1,sizeof(float)*H*groupSize,&get_reply,0,0,0);
          while(get_reply!=1);
          for(j=0;j<groupSize;++j) //control 4 lines
          {
              //gate_t[d]       : ldm_temp[j*4*H]
              //gate_t[d+H]     : ldm_temp[j*4*H + H]
              //gate_t[d+2*H]   : ldm_temp[j*4*H + 2*H]
              //gate_t[d+3*H]   : ldm_temp[j*4*H + 3*H]
              for(k=0;k<H;++k)
              {
                  ldm_c_t[j*H+k] = ldm_temp[j*4*H + H + k] * ldm_c_t_1[j*H+k] + ldm_temp[j*4*H + k] * ldm_temp[j*4*H + 3*H + k];
                  ldm_h_t[j*H+k] = ldm_temp[j*4*H + 2*H + k] * tanh(ldm_c_t[j*H+k]);
              }
          }
          put_reply=0;
          athread_put(PE_MODE,ldm_c_t,&c_t[i/4*H],sizeof(float)*H*groupSize,&put_reply,0,0);
          while(put_reply!=1);
          put_reply=0;
          athread_put(PE_MODE,ldm_h_t,&h_t[i/4*H],sizeof(float)*H*groupSize,&put_reply,0,0);
          while(put_reply!=1);
      }


  }


  ldm_free(ldm_pre_gate,BUFFSIZE);
  ldm_free(ldm_temp,BUFFSIZE);
  ldm_free(ldm_clip,2*1024);
}


