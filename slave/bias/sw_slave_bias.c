#include <stdio.h> 
#include <float.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include <simd.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
//#include "caffe/swlayers/bias_type.h"
#include "include/swbias.h"

#define min(a,b) ((a)>(b)?(b):(a))
#define max(a,b) ((a)>(b)?(a):(b))
#define SIMDSIZE 4
#define SIMDTYPEF floatv4

//__thread_local volatile unsigned long get_reply,put_reply;

void biasForward(SlaveBiasParam *pParam)
{
    int my_id;
    volatile unsigned long get_reply,put_reply;
    int dim,bias_dim,inner_dim,outer_dim;
    int numofline,line_id,channel_id,i,j,k;
    float * bottom_data;
    float * bias_data;
    float * top_data;
    SIMDTYPEF vsrc1,vsrc2,vsrc3,vsrc4;
    SIMDTYPEF vdst;
 
    my_id = athread_get_id(-1);
    dim=pParam->dim;
    bias_dim=pParam->bias_dim;
    inner_dim=pParam->inner_dim;
    outer_dim=pParam->outer_dim;
    bottom_data=pParam->bottom_data;
    top_data=pParam->top_data;
    bias_data=pParam->bias_data;

    float * ldm_line_raw = (float*)(long)ldm_malloc(inner_dim*4 + 32);
    float * ldm_line = ldm_line_raw + ((((((long)ldm_line_raw))%16)==0)?0:(16-(((long)ldm_line_raw))%16));

    float * ldm_bias = (float*)(long)ldm_malloc(bias_dim*4);
    get_reply=0;
    athread_get(PE_MODE,bias_data,ldm_bias,bias_dim*4,&get_reply,0,0,0);
    while(get_reply!=1);
    numofline=bias_dim*outer_dim;

    for(i=0;i<numofline;i+=64)
    {
        line_id=i+my_id;
        channel_id=line_id % bias_dim;
        if(line_id>=numofline) break;
        get_reply=0;
        athread_get(PE_MODE,&bottom_data[line_id*inner_dim],ldm_line,inner_dim*4,&get_reply,0,0,0);
        while(get_reply!=1);

        /*
        for(j=0;j<inner_dim;++j)
        {
            ldm_line[j]+=ldm_bias[channel_id];
        }
        */
        vsrc1=ldm_bias[channel_id];
        for(j=0;j<inner_dim;j+=SIMDSIZE)
        {
            simd_load(vsrc2,&ldm_line[j]);
            vdst=vsrc1+vsrc2;
            simd_store(vdst,&ldm_line[j]);
        }
        put_reply=0;
        athread_put(PE_MODE,ldm_line,&top_data[line_id*inner_dim],inner_dim*4,&put_reply,0,0);
        while(put_reply!=1);
    }
    ldm_free(ldm_line_raw,inner_dim*4+32);
    ldm_free(ldm_bias,bias_dim*4);
}



void biasBackward(SlaveBiasParam *pParam)
{
    volatile unsigned long get_reply,put_reply;
    int dim,bias_dim,inner_dim,outer_dim;
    int numofline,line_id,channel_id,arrayindex,i,j,k;
    float accum;
    float sum;
    float * bias_diff;
    float * top_diff;
    SIMDTYPEF vsrc1,vsrc2,vsrc3,vsrc4;
    SIMDTYPEF vdst;
 
    int my_id = athread_get_id(-1);

    accum=pParam->accum;
    dim=pParam->dim;
    bias_dim=pParam->bias_dim;
    inner_dim=pParam->inner_dim;
    outer_dim=pParam->outer_dim;
    bias_diff=pParam->bias_diff;
    top_diff=pParam->top_diff;

    float * ldm_line_raw = (float*)(long)ldm_malloc(inner_dim*4 + 32);
    float * ldm_line = ldm_line_raw + ((((((long)ldm_line_raw))%16)==0)?0:(16-(((long)ldm_line_raw))%16));

    float * ldm_sum=(float*)(long)ldm_malloc(bias_dim*4);
    
    numofline=bias_dim*outer_dim;

    for(i=0;i<bias_dim;++i)
    {
      ldm_sum[i]=0.0;
    }

    // for(i=0;i<numofline;i+=64)
    // {
    //     line_id=i+my_id;

    //     channel_id=line_id % bias_dim;
    //     if(line_id>=numofline) break;
    //     get_reply=0;
    //     athread_get(PE_MODE,&top_diff[line_id*inner_dim],ldm_line,inner_dim*4,&get_reply,0,0,0);
    //     while(get_reply!=1);

    //     sum=0.0;
    //     for(j=0;j<inner_dim;++j)
    //     {
    //         sum+=ldm_line[j];
    //     }
        
    //     bias_diff[channel_id]=accum*bias_diff[channel_id]+sum;
    //     if(line_id<bias_dim) accum=1.0;
    // }

    for(i=0;i<outer_dim;++i)
    {
        for(j=0;j<bias_dim;j+=64)
        {
            channel_id=j+my_id;
            if(channel_id>=bias_dim) break;
            arrayindex=j/64;
            line_id=i*bias_dim+channel_id;
            get_reply=0;
            athread_get(PE_MODE,&top_diff[line_id*inner_dim],ldm_line,4*inner_dim,&get_reply,0,0,0);
            while(get_reply!=1);
            for(k=0;k<inner_dim;++k)
            {
              ldm_sum[arrayindex]+=ldm_line[k];
              
            }
        }

        if(i==outer_dim-1)
        {
            for(j=0;j<bias_dim;j+=64)
            {
                channel_id=j+my_id;
                if(channel_id>=bias_dim) break;
                arrayindex=j/64;
                if(((accum-0.0)>0.001)||((accum-0.0)<-0.001)) bias_diff[channel_id]=accum*ldm_sum[arrayindex];
                else bias_diff[channel_id]+=accum*ldm_sum[arrayindex];
                
            }
        }
    }


    ldm_free(ldm_line_raw,inner_dim*4+32);
    ldm_free(ldm_sum,bias_dim*4);
}
