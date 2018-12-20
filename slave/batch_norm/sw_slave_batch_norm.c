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
//#include "caffe/swlayers/batch_norm_type.h"

#include "../../include/swbatchnorm.h"


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






void batch_slave_use_forward_f(BNData* param)
{

  int my_id;
  volatile unsigned long get_reply,put_reply;

	int num,channels,spatial_dim,num_of_line,line_id,i,j,divide_index;
	float eps,ldm_value,ldm_divide_value;

  SIMDTYPEF vsrc1,vsrc2,vsrc3,vsrc4;
  SIMDTYPEF vdst;

  float * bottom_data=param->bottom_data;
  float * top_data=param->top_data;
  float * mean_by_channel=param->mean_by_channel;
  float * variance_by_channel=param->variance_by_channel;
  float * temp_mutable=param->temp_mutable;
  float * xnorm=param->xnorm;
	num=param->num;
	channels=param->channels;
	spatial_dim=param->spatial_dim;
  eps=param->eps;
	num_of_line=num*channels;
  my_id=athread_get_id(-1);

  float * ldm_line_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_line = ldm_line_raw + ((((((long)ldm_line_raw))%16)==0)?0:(16-(((long)ldm_line_raw))%16));

  float * ldm_temp_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_temp = ldm_temp_raw + ((((((long)ldm_temp_raw))%16)==0)?0:(16-(((long)ldm_temp_raw))%16));

  //float * ldm_line =(float*)(long)ldm_malloc(spatial_dim*4);
  float * ldm_mean=(float*)(long)ldm_malloc(channels*4);
  float * ldm_variance=(float*)(long)ldm_malloc(channels*4);
  //float * ldm_temp=(float*)(long)ldm_malloc(spatial_dim*4);
  float * ldm_divide=(float*)(long)ldm_malloc(channels*4);
  float * ldm_mul=(float*)(long)ldm_malloc(channels*4);
  get_reply=0;
  athread_get(PE_MODE,mean_by_channel,ldm_mean,channels*4,&get_reply,0,0,0);
  while(get_reply!=1);
  get_reply=0;
  athread_get(PE_MODE,variance_by_channel,ldm_variance,channels*4,&get_reply,0,0,0);
  while(get_reply!=1);
  for(i=0;i<channels;++i)
  {
    ldm_divide[i]=sqrt(ldm_variance[i]+eps);
    ldm_mul[i]=1.0/ldm_divide[i];
    //soft_divide(1.0,ldm_divide[i]);
  }
	for(i=0;i<num_of_line;i+=64)
	{
		//i : line_index
    line_id=i+my_id;
    if(line_id>=num_of_line) break;
    get_reply=0;
    athread_get(PE_MODE,&bottom_data[line_id*spatial_dim],ldm_line,spatial_dim*4,&get_reply,0,0,0);
    while(get_reply!=1);
    //ldm_value=sqrt(ldm_variance[line_id%channels]+eps);
    //ldm_divide_value=1.0/ldm_value;
    divide_index=line_id%channels;


    // for(j=0;j<spatial_dim;++j)
    // {
    //   ldm_temp[j]=ldm_divide[divide_index];
    //   ldm_line[j]=(ldm_line[j]-ldm_mean[divide_index])*ldm_mul[divide_index];
    // }

    vsrc2=ldm_mean[divide_index];
    vsrc3=ldm_mul[divide_index];
    vsrc4=ldm_divide[divide_index];

    for(j=0;j<spatial_dim;j+=SIMDSIZE)
    {
      simd_load(vsrc1,&ldm_line[j]);
      vdst=(vsrc1-vsrc2)*vsrc3;
      simd_store(vsrc4,&ldm_temp[j]);
      simd_store(vdst,&ldm_line[j]);
    }


    put_reply=0;
    athread_put(PE_MODE,ldm_line,&top_data[line_id*spatial_dim],spatial_dim*4,&put_reply,0,0);
    while(put_reply!=1);
    put_reply=0;
    athread_put(PE_MODE,ldm_line,&xnorm[line_id*spatial_dim],spatial_dim*4,&put_reply,0,0);
    while(put_reply!=1);
    put_reply=0;
    athread_put(PE_MODE,ldm_temp,&temp_mutable[line_id*spatial_dim],spatial_dim*4,&put_reply,0,0);
    while(put_reply!=1);
	}
  ldm_free(ldm_line_raw,spatial_dim*4+32);
  ldm_free(ldm_mean,channels*4);
  ldm_free(ldm_variance,channels*4);
  ldm_free(ldm_temp_raw,spatial_dim*4+32);
  ldm_free(ldm_divide,channels*4);
  ldm_free(ldm_mul,channels*4);
}

void batch_slave_nouse_forward_f(BNData* param)
{

  int my_id;
  volatile unsigned long get_reply,put_reply;
	int num,channels,spatial_dim,num_of_line,line_id,channel_id,i,j,k,arrayindex;
  float ldm_value,eps;
  float * bottom_data=param->bottom_data;
  float * top_data=param->top_data;
  float * mean_by_channel=param->mean_by_channel;
  float * variance_by_channel=param->variance_by_channel;
  float * temp_mutable=param->temp_mutable;
  float * xnorm=param->xnorm;
	num=param->num;
	channels=param->channels;
	spatial_dim=param->spatial_dim;
	num_of_line=num*channels;
  my_id=athread_get_id(-1);
  eps=param->eps;

  float * ldm_line_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_line = ldm_line_raw + ((((((long)ldm_line_raw))%16)==0)?0:(16-(((long)ldm_line_raw))%16));

  float * ldm_temp_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_temp = ldm_temp_raw + ((((((long)ldm_temp_raw))%16)==0)?0:(16-(((long)ldm_temp_raw))%16));

  //float * ldm_line=(float*)(long)ldm_malloc(spatial_dim*4);
  float * ldm_mean=(float*)(long)ldm_malloc(channels*4);
  float * ldm_variance=(float*)(long)ldm_malloc(channels*4);
  //float * ldm_temp=(float*)(long)ldm_malloc(spatial_dim*4);
  float * ldm_divide=(float*)(long)ldm_malloc(channels*4);
  float * ldm_mul=(float*)(long)ldm_malloc(channels*4);

  SIMDTYPEF vsrc1,vsrc2,vsrc3,vsrc4;
  SIMDTYPEF vdst;


  //uint64_t pt0,pt1,pt2,pt3,pt4,pt99;

  for(i=0;i<channels;++i)
  {
    ldm_mean[i]=0.0;
    ldm_variance[i]=0.0;
  }

  //if(my_id==0) pt0=slave_get_rtc();

  for(i=0;i<num;++i)
  {
    for(j=0;j<channels;j+=64)
    {
      channel_id=j+my_id;
      if(channel_id>=channels) break;
      arrayindex=j/64;
      line_id=i*channels+channel_id;
      get_reply=0;
      athread_get(PE_MODE,&bottom_data[line_id*spatial_dim],ldm_line,4*spatial_dim,&get_reply,0,0,0);
      while(get_reply!=1);
      for(k=0;k<spatial_dim;++k)
      {
        ldm_mean[arrayindex]+=ldm_line[k];
        ldm_variance[arrayindex]+=(ldm_line[k]*ldm_line[k]);
      }
    }
  }


  //for(i=0;i<num;++i)
  //{
    for(j=0;j<channels;j+=64)
    {
      channel_id=j+my_id;
      if(channel_id>=channels) break;
      arrayindex=j/64;
  //    if(i==0)
  //    {
        ldm_mean[arrayindex]/=(spatial_dim*num);
        ldm_variance[arrayindex]/=(spatial_dim*num);
  //    }
  //    line_id=i*channels+channel_id;
      // get_reply=0;
      // athread_get(PE_MODE,&top_data[line_id*spatial_dim],ldm_line,4*spatial_dim,&get_reply,0,0,0);
      // while(get_reply!=1);

      // vsrc1=ldm_mean[arrayindex];
      // vsrc2=ldm_variance[arrayindex];
      // for(k=0;k<spatial_dim;k+=SIMDSIZE)
      // {
      //   // simd_load(vsrc2,&ldm_line[k]);
      //   vdst=vsrc2-vsrc1;
      //   simd_store(vdst,&ldm_line[k]);
      //   vdst=vdst*vdst;
      //   simd_store(vdst,&ldm_temp[k]);

      // }

      // for(k=0;k<spatial_dim;++k)
      // {
      //   ldm_variance[arrayindex]+=ldm_temp[k];
      // }

      ldm_variance[arrayindex]=ldm_variance[arrayindex]-ldm_mean[arrayindex]*ldm_mean[arrayindex];
    }
  // }

  //if(my_id==0) pt2=slave_get_rtc();

//if(my_id==0) printf("here 3\n");

  for(i=0;i<num;++i)
  {
    for(j=0;j<channels;j+=64)
    {
      channel_id=j+my_id;
      if(channel_id>=channels) break;
      arrayindex=j/64;
      if(i==0)
      {
        // ldm_variance[arrayindex]/=(spatial_dim*num);
        ldm_divide[arrayindex]=sqrt(ldm_variance[arrayindex]+eps);
        ldm_mul[arrayindex]=1.0/ldm_divide[arrayindex];
      }
      line_id=i*channels+channel_id;
      get_reply=0;
      athread_get(PE_MODE,&top_data[line_id*spatial_dim],ldm_line,4*spatial_dim,&get_reply,0,0,0);
      while(get_reply!=1);

      vsrc2=ldm_mean[arrayindex];
      vsrc3=ldm_mul[arrayindex];
      vsrc4=ldm_divide[arrayindex];
      for(k=0;k<spatial_dim;k+=SIMDSIZE)
      {
        simd_load(vsrc1,&ldm_line[k]);
        vdst=(vsrc1-vsrc2)*vsrc3;
        simd_store(vsrc4,&ldm_temp[k]);
        simd_store(vdst,&ldm_line[k]);
      }


      put_reply=0;
      athread_put(PE_MODE,ldm_line,&top_data[line_id*spatial_dim],spatial_dim*4,&put_reply,0,0);
      while(put_reply!=1);
      put_reply=0;
      athread_put(PE_MODE,ldm_line,&xnorm[line_id*spatial_dim],spatial_dim*4,&put_reply,0,0);
      while(put_reply!=1);
      put_reply=0;
      athread_put(PE_MODE,ldm_temp,&temp_mutable[line_id*spatial_dim],spatial_dim*4,&put_reply,0,0);
      while(put_reply!=1);
      mean_by_channel[channel_id]=ldm_mean[arrayindex];
      variance_by_channel[channel_id]=ldm_variance[arrayindex];
    }
  }

  //if(my_id==0) printf("here 4\n");
  //if(my_id==0) pt3=slave_get_rtc();
  //if(my_id==0)
  //{
  //  printf("compute mean=%f\n",(double)(pt1-pt0)/(1.45*1024*1024*1024));
  //  printf("compute var=%f\n",(double)(pt2-pt1)/(1.45*1024*1024*1024));
  //  printf("main step=%f\n",(double)(pt3-pt2)/(1.45*1024*1024*1024));
  //}

  ldm_free(ldm_line_raw,spatial_dim*4+32);
  ldm_free(ldm_mean,channels*4);
  ldm_free(ldm_variance,channels*4);
  ldm_free(ldm_temp_raw,spatial_dim*4+32);
  ldm_free(ldm_divide,channels*4);
  ldm_free(ldm_mul,channels*4);
}
void batch_slave_use_backward_f(BNData* param)
{
  int my_id;
  volatile unsigned long get_reply,put_reply;
	int num,channels,spatial_dim,num_of_line,line_id,i,j;
	float eps;
  float * bottom_diff=param->bottom_diff;
  float * top_diff=param->top_diff;
  float * temp_mutable=param->temp_mutable;
	num=param->num;
	channels=param->channels;
	spatial_dim=param->spatial_dim;
	num_of_line=num*channels;
  my_id=athread_get_id(-1);

  //float * ldm_line=(float*)(long)ldm_malloc(spatial_dim*4);
  //float * ldm_temp=(float*)(long)ldm_malloc(spatial_dim*4);
  float * ldm_line_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_line = ldm_line_raw + ((((((long)ldm_line_raw))%16)==0)?0:(16-(((long)ldm_line_raw))%16));

  float * ldm_temp_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_temp = ldm_temp_raw + ((((((long)ldm_temp_raw))%16)==0)?0:(16-(((long)ldm_temp_raw))%16));

  SIMDTYPEF vsrc1,vsrc2,vsrc3,vdst;

	for(i=0;i<num_of_line;i+=64)
	{
		//i : line_index
    line_id=i+my_id;
    if(line_id>=num_of_line) break;
    get_reply=0;
    athread_get(PE_MODE,&top_diff[line_id*spatial_dim],ldm_line,spatial_dim*4,&get_reply,0,0,0);
    while(get_reply!=1);
    get_reply=0;
    athread_get(PE_MODE,&temp_mutable[line_id*spatial_dim],ldm_temp,spatial_dim*4,&get_reply,0,0,0);
    while(get_reply!=1);
    //for(j=0;j<spatial_dim;++j)
    //{
    //  ldm_line[j]/=ldm_temp[j];
    //}
    for(j=0;j<spatial_dim;j+=SIMDSIZE)
    {
      simd_load(vsrc1,&ldm_line[j]);
      simd_load(vsrc2,&ldm_temp[j]);
      vdst=vsrc1/vsrc2;
      simd_store(vdst,&ldm_line[j]);
    }
    put_reply=0;
    athread_put(PE_MODE,ldm_line,&bottom_diff[line_id*spatial_dim],spatial_dim*4,&put_reply,0,0);
    while(put_reply!=1);
	}
  ldm_free(ldm_line_raw,spatial_dim*4+32);
  ldm_free(ldm_temp_raw,spatial_dim*4+32);
}

void batch_slave_nouse_backward_f(BNData* param)
{
  //printf("here 1\n");
  int my_id;
  volatile unsigned long get_reply,put_reply;

	int num,channels,spatial_dim,num_of_line,line_id,channel_id,i,j,k,arrayindex;
  float ldm_value;
  float * bottom_diff=param->bottom_diff;
  float * top_diff=param->top_diff;
  float * top_data=param->top_data;
  float * temp_mutable=param->temp_mutable;
	num=param->num;
	channels=param->channels;
	spatial_dim=param->spatial_dim;
	num_of_line=num*channels;
  my_id=athread_get_id(-1);
  //printf("here 2\n");
  //float * ldm_line=(float*)(long)ldm_malloc(spatial_dim*4);
  float * ldm_mean=(float*)(long)ldm_malloc(channels*4);
  float * ldm_variance=(float*)(long)ldm_malloc(channels*4);
  //float * ldm_temp=(float*)(long)ldm_malloc(spatial_dim*4);
  //float * ldm_data=(float*)(long)ldm_malloc(spatial_dim*4);
  //float * ldm_diff=(float*)(long)ldm_malloc(spatial_dim*4);
  float * ldm_alpha=(float*)(long)ldm_malloc(channels*4);
  float * ldm_beta=(float*)(long)ldm_malloc(channels*4);

  float * ldm_line_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_line = ldm_line_raw + ((((((long)ldm_line_raw))%16)==0)?0:(16-(((long)ldm_line_raw))%16));

  float * ldm_data_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_data = ldm_data_raw + ((((((long)ldm_data_raw))%16)==0)?0:(16-(((long)ldm_data_raw))%16));

  float * ldm_diff_raw = (float*)(long)ldm_malloc(spatial_dim*4 + 32);
  float * ldm_diff = ldm_diff_raw + ((((((long)ldm_diff_raw))%16)==0)?0:(16-(((long)ldm_diff_raw))%16));

  SIMDTYPEF vsrc1,vsrc2,vsrc3,vsrc4,vsrc5,vdst;

  for(i=0;i<channels;++i)
  {
    ldm_alpha[i]=0.0;
    ldm_beta[i]=0.0;
  }

  for(i=0;i<num;++i)
  {
    for(j=0;j<channels;j+=64)
    {
      channel_id=j+my_id;
      if(channel_id>=channels) break;
      arrayindex=j/64;
      line_id=i*channels+channel_id;
      get_reply=0;
      athread_get(PE_MODE,&top_diff[line_id*spatial_dim],ldm_diff,4*spatial_dim,&get_reply,0,0,0);
      while(get_reply!=1);
      get_reply=0;
      athread_get(PE_MODE,&top_data[line_id*spatial_dim],ldm_data,4*spatial_dim,&get_reply,0,0,0);
      while(get_reply!=1);
      for(k=0;k<spatial_dim;++k)
      {
        ldm_alpha[arrayindex]+=ldm_diff[k];
        ldm_beta[arrayindex]+=(ldm_diff[k]*ldm_data[k]);
      }
    }
  }

  for(i=0;i<num;++i)
  {
    for(j=0;j<channels;j+=64)
    {
      channel_id=j+my_id;
      if(channel_id>=channels) break;
      arrayindex=j/64;
      if(i==0)
      {
        ldm_alpha[arrayindex]/=(spatial_dim*num);
        ldm_beta[arrayindex]/=(spatial_dim*num);
      }
      line_id=i*channels+channel_id;
      get_reply=0;
      athread_get(PE_MODE,&top_diff[line_id*spatial_dim],ldm_diff,4*spatial_dim,&get_reply,0,0,0);
      while(get_reply!=1);
      get_reply=0;
      athread_get(PE_MODE,&top_data[line_id*spatial_dim],ldm_data,4*spatial_dim,&get_reply,0,0,0);
      while(get_reply!=1);
      //get_reply=0;
      //athread_get(PE_MODE,&temp_mutable[line_id*spatial_dim],ldm_temp,4*spatial_dim,&get_reply,0,0,0);
      //while(get_reply!=1);
      ldm_value=1.0/temp_mutable[line_id*spatial_dim];
      //for(k=0;k<spatial_dim;++k)
      //{
      //  ldm_line[k]=(ldm_diff[k]-ldm_alpha[arrayindex]-ldm_data[k]*ldm_beta[arrayindex])*ldm_value;
      //}
      vsrc3=ldm_alpha[arrayindex];
      vsrc4=ldm_beta[arrayindex];
      vsrc5=ldm_value;
      for(k=0;k<spatial_dim;k+=SIMDSIZE)
      {
        simd_load(vsrc1,&ldm_diff[k]);
        simd_load(vsrc2,&ldm_data[k]);
        vdst=(vsrc1-vsrc3-vsrc2*vsrc4)*vsrc5;
        simd_store(vdst,&ldm_line[k]);
      }
      put_reply=0;
      athread_put(PE_MODE,ldm_line,&bottom_diff[line_id*spatial_dim],4*spatial_dim,&put_reply,0,0);
      while(put_reply!=1);
    }
  }
  ldm_free(ldm_line_raw,spatial_dim*4+32);
  ldm_free(ldm_mean,channels*4);
  ldm_free(ldm_variance,channels*4);
  //ldm_free(ldm_temp,spatial_dim*4);
  ldm_free(ldm_data_raw,spatial_dim*4+32);
  ldm_free(ldm_diff_raw,spatial_dim*4+32);
  ldm_free(ldm_alpha,channels*4);
  ldm_free(ldm_beta,channels*4);
}


