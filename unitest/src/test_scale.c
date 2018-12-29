#include "include/swscale.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
int test_scale() 
{
    int a[3],b[5],cc[4];
    a[0]=32;a[1]=64;a[2]=128;
    b[0]=3;b[1]=64;b[2]=128;b[3]=256;b[4]=512;
    cc[0]=6;cc[1]=14;cc[2]=28;cc[3]=56;
    int num,channels_,width_,height_;
    int spatial_dim;
    int i,j,k,z,ph,pw,n,c,h,w;
    int blob_size;
    int ii,jj,kk;
    float eps_=1e-4;
    float factor;


    float * bbottom_data=(float*)malloc(sizeof(float)*128*512*56*56);
    float * sscale_data=(float*)malloc(sizeof(float)*128*512*56*56);

    //without origin data
    float * top_data=(float*)malloc(sizeof(float)*128*512*56*56);
    float * my_top_data=(float*)malloc(sizeof(float)*128*512*56*56);

    char out[20]="0 0 0 time";

    athread_init();
    printf("start\n");
    for(ii=0;ii<3;++ii)
    {
      for(jj=0;jj<5;++jj)
      {
        for(kk=0;kk<4;++kk)
        {
            //if(k==1) break;
            printf("i=%d j=%d k=%d\n",ii,jj,kk);
            num=a[ii];channels_=b[jj];width_=height_=cc[kk];
            if(ii==2&&jj==4&&kk==3)
            {
              num=1;channels_=128*512;
            }
            spatial_dim=width_*height_;
            blob_size=num*channels_*width_*height_;
            //with origin data
            //printf("here 1\n");
            //init
            for(z = 0; z < blob_size; ++z ) bbottom_data[z] = rand()/(float)RAND_MAX;
            for(z = 0; z < blob_size; ++z ) sscale_data[z] = rand()/(float)RAND_MAX;

            out[0]='0'+ii;
            out[2]='0'+jj;
            out[4]='0'+kk;


            int mode = 0;
            //mode==0 : max
            //mode==1 : avg

            int use_top_mask = 0;
            int global_pooling_ = 0;

            int kernel_h_= 2;
            int kernel_w_= 2;
            int pooled_height_;
            int pooled_width_;
            int stride_h_=2;
            int stride_w_=2;
            int pad_h_=0;
            int pad_w_=0;





  for(i=0;i<num;++i)
  {
    for(j=0;j<channels_;++j)
    {
      factor = sscale_data[j];
      for(k=0;k<spatial_dim;++k)
      {
        my_top_data[i*channels_*spatial_dim+j*spatial_dim+k]=factor*bbottom_data[i*channels_*spatial_dim+j*spatial_dim+k];
      }
    }
  }




  sw_scale_layer_f(
      (float*)bbottom_data,
      (float*)sscale_data,
      (float*)top_data,
      num,
      spatial_dim,
      channels_
      );



   int flag=1;
   for(i=0;i<blob_size;++i)
   {
     if(top_data[i]-my_top_data[i]>0.0001||
        top_data[i]-my_top_data[i]<-0.0001)
     {
       flag=0;
       printf("error n=%d c=%d top_data[%d]=%f my_top_data[%d]=%f\n",num,channels_,i,top_data[i],i,my_top_data[i]);
     }
   }
   if(flag) printf("ok\n");
   else exit(0);

        }
      }
    }


     athread_halt();

        free(bbottom_data);
        free(sscale_data);
        free(top_data);
        free(my_top_data);

  return 0;
}

