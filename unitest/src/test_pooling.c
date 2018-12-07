#include <math.h>
#include <stdio.h>
#include "athread.h"
#include <sys/time.h>
#define Dtype float
#include <stdlib.h>
#include <string.h>
#include "../../include/swpool.h"
#include "include/swcommon.h"
#define MAXTIMERSIZE 1000

#define NUM_THREADS 64

// extern void SLAVE_FUN(poolingForwardMax_f)();
// extern void SLAVE_FUN(poolingForwardAvg_f)();

int timer[MAXTIMERSIZE];
int timer_num = 0;
int isTiming = 0;
long startTime;
long startSec;
long startUsec;
char* timer_name[MAXTIMERSIZE];
int timer_index;
/*
typedef struct _tagSlavePoolingParam_f
{
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nThreadsNum,nLeftThreadsNum;
	int nBottomOffset,nTopOffset,use_top_mask;
	int  *pMask;
	float *pTopData,*pBottomData,*pTopMask;
}SlavePoolingParam_f;
*/

void caffe_set_float(const int N, const Dtype alpha, Dtype* Y) {
  int i;
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  
    return;
  }
  for (i = 0; i < N; ++i) {
    Y[i] = alpha; 
  }
  }

void caffe_set_int(const int N, const int alpha, int* Y) {
  int i;
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  
    return;
  }
  for (i = 0; i < N; ++i) {
    Y[i] = alpha; 
  }
  }


inline int offset(int channel, int height, int width,  int n,  int c )  
  {
    return ((n * channel + c) * height ) * width ;
  }

int min(int a,int b) {return a>b?b:a;}
int max(int a,int b) {return a>b?a:b;}

int pooling_judge_small(int N, int C,int height_, int width_)
{
	if(height_ * width_ < 28 * 28 && N * C >= 64) return 1;
	return -1;
}

int test_pooling()
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

    struct timeval t1,t2;

    float * bbottom_data=(float*)malloc(sizeof(float)*128*512*56*56);

    //without origin data
    float * top_0=(float*)malloc(sizeof(float)*128*512*56*56);
    float * top_1=(float*)malloc(sizeof(float)*128*512*56*56);
    int * maxidx=(int*)malloc(sizeof(int)*128*512*56*56);
    //float * variance_origin=(float*)malloc(sizeof(float)*512);
    //float * xnorm=(float*)malloc(sizeof(float)*128*512*56*56);
    //float * temp=(float*)malloc(sizeof(float)*128*512*56*56);
    float * my_top_0=(float*)malloc(sizeof(float)*128*512*56*56);
    float * my_top_1=(float*)malloc(sizeof(float)*128*512*56*56);
    int * my_maxidx=(int*)malloc(sizeof(int)*128*512*56*56);

    char out[20]="0 0 0 time";

    printf("start\n");
    for(ii=0;ii<3;++ii)
    {
      for(jj=0;jj<5;++jj)
      {
        for(kk=0;kk<4;++kk)
        {
            //if(k==1) break;
            double pooling_time=0.0;
            printf("i=%d j=%d k=%d\n",ii,jj,kk);
            num=a[ii];channels_=b[jj];width_=height_=cc[kk];
            spatial_dim=width_*height_;
            blob_size=num*channels_*width_*height_;
            //with origin data
            //printf("here 1\n");
            //init
            //for(z = 0; z < blob_size; ++z ) bottom_data[z] = rand()/(float)RAND_MAX;
            for(z = 0; z < blob_size; ++z ) bbottom_data[z] = rand()/(float)RAND_MAX;
            //for(z = 0; z < blob_size; ++z ) top_diff[z] = rand()/(float)RAND_MAX;
            //for(z = 0; z < blob_size; ++z ) bottom_diff[z] = rand()/(float)RAND_MAX;
            //for(z = 0; z < channels; ++z ) blob_0[z] = rand()/(float)RAND_MAX;
            //for(z = 0; z < channels; ++z ) blob_1[z] = rand()/(float)RAND_MAX;
            //for(z = 0; z < 1; ++z ) blob_2[z] = rand()/(float)RAND_MAX;
            //for(z = 0; z < spatial_dim; ++z ) spatial_sum_multiplier[z] = 1.0;
            //for(z = 0; z < num; ++z ) batch_sum_multiplier[z] = 1.0;
            //for(z = 0; z < num*channels; ++z ) num_by_chans[z] = 1.0;

            //for(z = 0; z < blob_size; ++z ) temp[z] = 1.0;
            //for(z = 0; z < channels; ++z ) mean_origin[z] = 1.0;
            //for(z = 0; z < channels; ++z ) variance_origin[z] = 1.0;

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

            //printf("here 2\n");
            if (global_pooling_) {
              kernel_h_ = height_;
              kernel_w_ = width_;
            }
            pooled_height_ = (int)(ceil((float)(
                height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
            pooled_width_ = (int)(ceil((float)(
                width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
            if (pad_h_ || pad_w_) {
              // If we have padding, ensure that the last pooling starts strictly
              // inside the image (instead of at the padding); otherwise clip the last.
              if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
               --pooled_height_;
              }
              if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
                --pooled_width_;
              }
            }
            //printf("here 3\n");



#ifdef USE_BLAS 



  const Dtype* my_bottom_data = bbottom_data;
  Dtype* my_top_data = my_top_0;
  const int top_count = blob_size/4;
  // We'll output the mask to top[1] if it's of size >1.
  //const bool use_top_mask = top.size() > 1;

  

  int* my_mask = NULL;  // suppress warnings about uninitialized variables
  Dtype* my_top_mask = NULL;


  //printf("here 4\n");
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (mode) {
  case 0:  //max
    // Initialize
    if (use_top_mask) {
      my_top_mask = my_top_1;
      caffe_set_float(top_count, -1.0, my_top_mask);
    } else {
      my_mask = my_maxidx;
      caffe_set_int(top_count, -1, my_mask);
    }
    caffe_set_float(top_count, -1e30, my_top_data);
    //printf("here 5\n");
    // The main loop
    for (n = 0; n < num; ++n) {
      for (c = 0; c < channels_; ++c) {
        //if(i==2&&j==3&&k==3) printf("batch=%d channels_=%d\n",n,c);
        for (ph = 0; ph < pooled_height_; ++ph) {
          for (pw = 0; pw < pooled_width_; ++pw) {
            //printf("gg\n");
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            //if(i==2&&j==3&&k==3) printf("ggg\n");
            const int pool_index = ph * pooled_width_ + pw;
            for (h = hstart; h < hend; ++h) {
              for (w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;

                //if(n==0&&c==0&&ph==0&&pw==0) printf("my_bottom_data[%d]=%f  my_top_data[%d]=%f\n",index,my_bottom_data[index],pool_index,my_top_data[pool_index]);

                if (my_bottom_data[index] > my_top_data[pool_index]) {
                  my_top_data[pool_index] = my_bottom_data[index];
                  if (use_top_mask) {
                    my_top_mask[pool_index] = (float)(index);
                  } else {
                    my_mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        // bottom_data += bottom[0]->offset(0, 1);
        // top_data += top[0]->offset(0, 1);
        //if(i==2&&j==3&&k==3) printf("out\n");
        //
        //if(n==0&&c==0) printf("bo of = %d\n",offset(channels_,height_,width_,0,1));
        //if(n==0&&c==0) printf("to fo = %d\n",offset(channels_,pooled_height_,pooled_width_,0,1));

        my_bottom_data += offset(channels_, height_, width_, 0, 1);
        my_top_data += offset(channels_, pooled_height_, pooled_width_, 0, 1);
        if (use_top_mask) {
          //top_mask += top[0]->offset(0, 1);
          my_top_mask += offset(channels_, pooled_height_, pooled_width_, 0, 1);
        } else {
          //mask += top[0]->offset(0, 1);
          my_mask += offset(channels_, pooled_height_, pooled_width_, 0, 1);
        }
      }
    }
    //printf("here 6\n");
    break;
  case 1: //avg
    for (i = 0; i < top_count; ++i) {
      my_top_data[i] = 0;
    }
    // The main loop
    for (n = 0; n < num; ++n) {
      for (c = 0; c < channels_; ++c) {
        for (ph = 0; ph < pooled_height_; ++ph) {
          for (pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (h = hstart; h < hend; ++h) {
              for (w = wstart; w < wend; ++w) {
                my_top_data[ph * pooled_width_ + pw] +=
                    my_bottom_data[h * width_ + w];
              }
            }
            my_top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        
        // bottom_data += bottom[0]->offset(0, 1);
        // top_data += top[0]->offset(0, 1);
        my_top_data += offset(channels_, pooled_height_, pooled_width_, 0, 1);
        my_bottom_data += offset(channels_, height_, width_, 0, 1);
      }
    }
    break;
  case 2:
    //NOT_IMPLEMENTED
    ;
    break;
  default:
    //LOG(FATAL) << "Unknown pooling method."
    ;
  }
  

#endif


#ifdef USE_SWDNN
  
  const Dtype* bottom_data = bbottom_data;
  Dtype* top_data = top_0;
  //const int top_count = blob_size/4;
  int* mask = NULL;  
  Dtype* top_mask = NULL;

switch (mode) {
  case 0: 
  //printf("begin caffe\n");


    if (use_top_mask) {
      top_mask = top_1;
      //caffe_set_float(top_count, -1.0, top_mask);
    } else {
      mask = maxidx;
      //caffe_set_int(top_count, -1, mask);
    }
    //caffe_set_float(top_count, -1e30, top_data);
  gettimeofday(&t1, NULL);
  if(pooling_judge_small(num, channels_, height_, width_) > 0)
  {
      pooling_forward_small_max_f(num,channels_,(float*)top_data,(const float*)bottom_data,(int*)mask,(float*)top_mask,offset(channels_, height_, width_, 0, 1),
  				offset(channels_, pooled_height_, pooled_width_, 0, 1),use_top_mask,pooled_height_, pooled_width_, stride_h_,
  				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);

  }
  else
  {
  pooling_forward_max_f(num,channels_,(float*)top_data,(const float*)bottom_data,(int*)mask,(float*)top_mask,offset(channels_, height_, width_, 0, 1),
  				offset(channels_, pooled_height_, pooled_width_, 0, 1),use_top_mask,pooled_height_, pooled_width_, stride_h_,
  				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);
  }
  gettimeofday(&t2, NULL);
  break;

  case 1:

  pooling_forward_avg_f(num,channels_,(float*)top_data,(const float*)bottom_data,offset(channels_, height_, width_, 0, 1),
				offset(channels_, pooled_height_, pooled_width_, 0, 1),pooled_height_, pooled_width_, stride_h_,
				stride_w_, pad_h_, pad_w_, kernel_h_, kernel_w_, height_, width_);
  break;
  default:
  ;
}

  
#endif


#ifdef USE_ALL 
   int flag=1;
   for(i=0;i<top_count;++i)
   {
     if(top_data[i]-my_top_0[i]>0.0001||
        top_data[i]-my_top_0[i]<-0.0001)
     {
       flag=0;
       printf("error n=%d c=%d top_data[%d]=%f my_top_data[%d]=%f\n",num,channels_,i,top_data[i],i,my_top_0[i]);
     }
   }
   if(flag) printf("ok\n");
   else exit(0);

   pooling_time = TIME(t1,t2);
   double total_data_size=num*channels_*width_*height_*sizeof(float)*1.5;
   printf("pooling_layer %dx%dx%dx%d : Bandwidth : %f GB/s, time : %f sec\n", num, channels_, width_, height_, total_data_size/1e9/pooling_time, pooling_time);



#endif
        }
      }
    }

    

        free(bbottom_data);
        //free(top_data);
        //free(top_diff);
        free(top_0);
        free(top_1);
        free(maxidx);
        free(my_top_0);
        free(my_top_1);
        free(my_maxidx);
        //free(blob_2);
        //free(mean_origin);
        //free(variance_origin);
        //free(spatial_sum_multiplier);
        //free(num_by_chans);
        //free(batch_sum_multiplier);
        //free(xnorm);
        //free(temp);

  //athread_init();


  //athread_halt();
  return 0;
}
