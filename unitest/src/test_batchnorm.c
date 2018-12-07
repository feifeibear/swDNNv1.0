#include "include/swbias.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define Dtype float
int test_batchnorm()
{
  int a[3],b[5],c[4];
  a[0]=32;a[1]=64;a[2]=128;
  b[0]=3;b[1]=64;b[2]=128;b[3]=256;b[4]=512;
  //c[0]=56;c[1]=28;c[2]=14;c[3]=7;
  c[0]=7;c[1]=14;c[2]=28;c[3]=56;
  int num,channels,w,h;
  int spatial_dim;
  int i,j,k,z;
  int blob_size;
  int use_global_stats_=0;
  float moving_average_fraction_=0.9;
  float eps_=1e-4;
  struct timeval t1,t2;


        float * bottom_data=(float*)malloc(sizeof(float)*128*512*56*56);
        float * top_data=(float*)malloc(sizeof(float)*128*512*56*56);
        float * blob_0=(float*)malloc(sizeof(float)*512);
        float * blob_1=(float*)malloc(sizeof(float)*512);
        float * blob_2=(float*)malloc(sizeof(float));
        float * spatial_sum_multiplier=(float*)malloc(sizeof(float)*56*56);
        float * num_by_chans=(float*)malloc(sizeof(float)*128*512);
        float * batch_sum_multiplier=(float*)malloc(sizeof(float)*128);

        //without origin data
        float * mean_origin=(float*)malloc(sizeof(float)*512);
        float * variance_origin=(float*)malloc(sizeof(float)*512);
        float * xnorm=(float*)malloc(sizeof(float)*128*512*56*56);
        float * temp=(float*)malloc(sizeof(float)*128*512*56*56);

        char out[20]="0 0 0 time";

  printf("start\n");
  for(i=0;i<3;++i){
    for(j=0;j<5;++j){
      for(k=0;k<4;++k)
      {
        //if(k==1) break;
        printf("i=%d j=%d k=%d\n",i,j,k);
        num=a[i];channels=b[j];w=h=c[k];
        spatial_dim=w*h;
        blob_size=num*channels*w*h;
        //with origin data
        //init
        for(z = 0; z < blob_size; ++z ) bottom_data[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < blob_size; ++z ) top_data[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < channels; ++z ) blob_0[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < channels; ++z ) blob_1[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < 1; ++z ) blob_2[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < spatial_dim; ++z ) spatial_sum_multiplier[z] = 1.0;
        for(z = 0; z < num; ++z ) batch_sum_multiplier[z] = 1.0;
        for(z = 0; z < num*channels; ++z ) num_by_chans[z] = 1.0;

        for(z = 0; z < blob_size; ++z ) temp[z] = 1.0;
        for(z = 0; z < channels; ++z ) mean_origin[z] = 1.0;
        for(z = 0; z < channels; ++z ) variance_origin[z] = 1.0;

        out[0]='0'+i;
        out[2]='0'+j;
        out[4]='0'+k;

        double bn_time=0.0;
        gettimeofday(&t1, NULL);
          if(use_global_stats_)
    {


        if(sizeof(Dtype)==sizeof(float)){
          sw_batch_norm_use_forward_impl_f(
              (float*)bottom_data,
              (float*)top_data,
              (float*)blob_0,
              (float*)blob_1,
              (float*)blob_2,
              (float*)mean_origin,
              (float*)variance_origin,
              (float*)temp,
              (float*)xnorm,
              eps_,
              num,         //batch_size
              channels,   //C
              spatial_dim  //H*W
          );
        }else{
          // sw_batch_norm_use_forward_impl_d(
          //     (double*)bottom_data,
          //     (double*)top_data,
          //     (double*)this->blobs_[0]->mutable_cpu_data(),
          //     (double*)this->blobs_[1]->mutable_cpu_data(),
          //     (double*)this->blobs_[2]->mutable_cpu_data(),
          //     (double*)mean_.mutable_cpu_data(),
          //     (double*)variance_.mutable_cpu_data(),
          //     (double*)temp_.mutable_cpu_data(),
          //     eps_,
          //     num,         //batch_size
          //     channels_,   //C
          //     spatial_dim  //H*W
          // );
        }
    }
    else
    {
        if(sizeof(Dtype)==sizeof(float)){
        sw_batch_norm_nouse_forward_impl_f(
            (float*)bottom_data,
            (float*)top_data,
            (float*)num_by_chans,
            (float*)temp,
            (float*)blob_0,
            (float*)blob_1,
            (float*)blob_2,
            (float*)(&moving_average_fraction_),
            (float*)xnorm,
            eps_,
            num,
            channels,
            spatial_dim
        );
        }
        else{
        // sw_batch_norm_nouse_forward_impl_d(
        //     (double*)bottom_data,
        //     (double*)top_data,
        //     (double*)num_by_chans_.mutable_cpu_data(),
        //     (double*)temp_.mutable_cpu_data(),
        //     (double*)this->blobs_[0]->mutable_cpu_data(),
        //     (double*)this->blobs_[1]->mutable_cpu_data(),
        //     (double*)this->blobs_[2]->mutable_cpu_data(),
        //     (double*)(&moving_average_fraction_),
        //     eps_,
        //     num,
        //     channels_,
        //     spatial_dim
        // );
        }

    }

      gettimeofday(&t2, NULL);
      bn_time = TIME(t1,t2);
      double total_data_size=num*channels*w*h*5*sizeof(float);
      printf("bn_layer %dx%dx%dx%d : Bandwidth : %f GB/s, time : %f sec\n", num, channels, w, h, total_data_size/1e9/bn_time, bn_time);

      }
    }
  }

  free(bottom_data);
        free(top_data);
        free(blob_0);
        free(blob_1);
        free(blob_2);
        free(mean_origin);
        free(variance_origin);
        free(spatial_sum_multiplier);
        free(num_by_chans);
        free(batch_sum_multiplier);
        free(xnorm);
        free(temp);

  //athread_init();


  //athread_halt();

  return 0;
}

