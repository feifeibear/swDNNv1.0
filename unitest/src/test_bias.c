#include "include/swbias.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int test_bias() 
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


    float * bbottom_data=(float*)malloc(sizeof(float)*128*512*56*56);
    float * bbias_data=(float*)malloc(sizeof(float)*512);
    //float * blob_0=(float*)malloc(sizeof(float)*512);
    //float * blob_1=(float*)malloc(sizeof(float)*512);
    //float * blob_2=(float*)malloc(sizeof(float));
    //float * spatial_sum_multiplier=(float*)malloc(sizeof(float)*56*56);
    //float * num_by_chans=(float*)malloc(sizeof(float)*128*512);
    //float * batch_sum_multiplier=(float*)malloc(sizeof(float)*128);

    //without origin data
    float * top_data=(float*)malloc(sizeof(float)*128*512*56*56);
    //float * top_1=(float*)malloc(sizeof(float)*128*512*56*56);
    //int * maxidx=(int*)malloc(sizeof(int)*128*512*56*56);
    //float * variance_origin=(float*)malloc(sizeof(float)*512);
    //float * xnorm=(float*)malloc(sizeof(float)*128*512*56*56);
    //float * temp=(float*)malloc(sizeof(float)*128*512*56*56);
    float * my_top_data=(float*)malloc(sizeof(float)*128*512*56*56);
    //float * my_top_1=(float*)malloc(sizeof(float)*128*512*56*56);
    float * multi=(int*)malloc(sizeof(float)*56*56);

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
            spatial_dim=width_*height_;
            blob_size=num*channels_*width_*height_;
            //with origin data
            //printf("here 1\n");
            //init
            for(z = 0; z < blob_size; ++z ) top_data[z] = 0.0;
            for(z = 0; z < blob_size; ++z ) my_top_data[z]=0.0;
            for(z = 0; z < blob_size; ++z ) bbottom_data[z] = rand()/(float)RAND_MAX;
            for(z = 0; z < channels_; ++z ) bbias_data[z] = rand()/(float)RAND_MAX;
            for(z = 0; z < spatial_dim; ++z ) multi[z] = 1.0;

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

           



#ifdef USE_BLAS 


  float * out_data=my_top_data;
  
  // We'll output the mask to top[1] if it's of size >1.
  //const bool use_top_mask = top.size() > 1;

  //caffe_copy(blob_size, bbottom_data, my_top_data);
  memcpy(my_top_data,bbottom_data,blob_size*sizeof(float));
  for (n = 0; n < num; ++n) {
    /*
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels_,
        spatial_dim, 1, 1.0, bbias_data,
        multi, 1.0, my_top_data);
    */
    for(i=0;i<channels_;++i)
    {
      for(j=0;j<spatial_dim;++j)
      {
        my_top_data[i*spatial_dim+j]+=bbias_data[i];
      }
    }
    my_top_data += channels_*spatial_dim;
  }

#endif


#ifdef USE_SWDNN
  
  sw_bias_impl_f((float *)bbottom_data,
                   (float *)top_data,
                   (float *)bbias_data,
                   channels_*spatial_dim,
                   channels_,
                   spatial_dim,
                   num);


  
#endif


#ifdef USE_ALL 
   int flag=1;
   for(i=0;i<100;++i)
   {
     if(top_data[i]-out_data[i]>0.0001||
        top_data[i]-out_data[i]<-0.0001)
     {
       flag=0;
       printf("error n=%d c=%d top_data[%d]=%f my_top_data[%d]=%f\n",num,channels_,i,top_data[i],i,out_data[i]);
     }
   }
   if(flag) printf("ok\n");
   else exit(0);

#endif
        }
      }
    }


     athread_halt();

        free(bbottom_data);
        free(bbias_data);
        free(top_data);
        free(multi);
        free(my_top_data);

  return 0;
}

