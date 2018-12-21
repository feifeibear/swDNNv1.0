#include "include/swsoftmax.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define Dtype float
int test_softmax()
{
  int a[3],b[5],c[4];
  a[0]=32;a[1]=64;a[2]=128;
  b[0]=3;b[1]=64;b[2]=128;b[3]=256;b[4]=512;
  c[0]=7;c[1]=14;c[2]=28;c[3]=56;
  int num,channels,w,h;
  int spatial_dim;
  int i,j,k,z;
  int ii,jj,kk;
  int blob_size;
  int use_global_stats_=0;
  float moving_average_fraction_=0.9;
  float eps_=1e-4;
  struct timeval t1,t2;
  int outer_num_,inner_num_,dim;
  float sum;

  float * top_diff=(float*)malloc(sizeof(float)*128*512*56*56);
  float * top_data=(float*)malloc(sizeof(float)*128*512*56*56);
  float * bottom_diff=(float*)malloc(sizeof(float)*128*512*56*56);
  float * scale_data=(float*)malloc(sizeof(float)*56*56);

  float * my_bottom_diff=(float*)malloc(sizeof(float)*128*512*56*56);
  float * my_scale_data=(float*)malloc(sizeof(float)*56*56);

  char out[20]="0 0 0 time";

  printf("start\n");
  for(ii=0;ii<3;++ii){
    for(jj=0;jj<5;++jj){
      for(kk=0;kk<4;++kk)
      {
        //if(k==1) break;
        printf("i=%d j=%d k=%d\n",ii,jj,kk);
        num=a[ii];channels=b[jj];w=h=c[kk];
        spatial_dim=w*h;
        blob_size=num*channels*w*h;

        outer_num_=num;
        inner_num_=spatial_dim;
        dim=channels*spatial_dim;
        //with origin data
        //init
        for(z = 0; z < blob_size; ++z ) top_data[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < blob_size; ++z ) top_diff[z] = rand()/(float)RAND_MAX;


        out[0]='0'+ii;
        out[2]='0'+jj;
        out[4]='0'+kk;

        /*
        caffe_copy(blob_size, top_diff, my_bottom_diff);
        for (int i = 0; i < outer_num_; ++i) {
          for (int k = 0; k < inner_num_; ++k) {
            my_scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
                my_bottom_diff + i * dim + k, inner_num_,
                top_data + i * dim + k, inner_num_);
          }
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
              -1., sum_multiplier_.cpu_data(), my_scale_data, 1., my_bottom_diff + i * dim);
        }
        caffe_mul(blob_size, my_bottom_diff, top_data, my_bottom_diff);
        */
        memcpy(my_bottom_diff, top_diff, blob_size*sizeof(float));
        for(i=0; i<outer_num_; ++i)
        {
          for(k=0; k<inner_num_; ++k)
          {
            sum = 0.0;
            for(j=0; j<channels; ++j)
            {
              sum += my_bottom_diff[i*dim + k + j*inner_num_] * top_data[i*dim + k + j*inner_num_];
            }
            my_scale_data[k] = sum;
          }
          for(j=0;j<channels;++j)
          {
            for(k=0;k<inner_num_;++k)
            {
              my_bottom_diff[i*dim+j*inner_num_+k] -= my_scale_data[k];
            }
          }
        }

        for(i=0;i<outer_num_;++i)
        {
          for(j=0;j<channels;++j)
          {
            for(k=0;k<inner_num_;++k)
            {
              my_bottom_diff[i*dim+j*inner_num_+k] *= top_data[i*dim+j*inner_num_+k];
            }
          }
        }

        double softmax_time=0.0;
        gettimeofday(&t1, NULL);
        sw_softmax_backward_impl_f(
          (float*)top_diff,
          (float*)top_data,
          (float*)bottom_diff,
          (float*)scale_data,
          outer_num_,
          channels,
          inner_num_,
          dim
            );

        gettimeofday(&t2, NULL);
        softmax_time = TIME(t1,t2);
        double total_data_size=num*channels*w*h*3*sizeof(float);
        int flag=1;
        for(i=0;i<outer_num_;++i)
        {
          for(j=0;j<channels;++j)
          {
            for(k=0;k<inner_num_;++k)
            {
              if(bottom_diff[i*dim+j*inner_num_+k]-my_bottom_diff[i*dim+j*inner_num_+k]>1e-4
              || bottom_diff[i*dim+j*inner_num_+k]-my_bottom_diff[i*dim+j*inner_num_+k]<-1e-4)
              {
                flag=0;
                printf("softmax backward error bottom_diff %d %d %d = %f my_bottom_diff %d %d %d = %f\n",i,j,k,bottom_diff[i*dim+j*inner_num_+k],i,j,k,my_bottom_diff[i*dim+j*inner_num_+k]);
                printf("top_diff = %f\ntop_data = %f\nscale_data = %f\n",top_diff[i*dim+j*inner_num_+k],top_data[i*dim+j*inner_num_+k],my_scale_data[k]);
                printf("top diff %f %f %f\n",top_diff[1],top_diff[50],top_diff[99]);
                printf("top data %f %f %f\n",top_data[1],top_data[50],top_data[99]);
                exit(0);
              }
            }
          }
        }
        if(flag) printf("check ok\n");
        printf("softmax_layer %dx%dx%dx%d : Bandwidth : %f GB/s, time : %f sec\n", num, channels, w, h, total_data_size/1e9/softmax_time, softmax_time);

      }
    }
  }
  free(top_diff);
  free(top_data);
  free(bottom_diff);
  free(scale_data);
  free(my_bottom_diff);
  free(my_scale_data);

  //athread_init();


  //athread_halt();

  return 0;
}

