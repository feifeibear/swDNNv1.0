#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>
// #include "caffe/swlayers/sw_batch_norm_layer_impl.h"
// #include "caffe/util/matrix_trans.h"
// #include "caffe/swlayers/batch_norm_type.h"

#include "../include/swbatchnorm.h"
extern SLAVE_FUN(batch_slave_use_forward_f)();
extern SLAVE_FUN(batch_slave_nouse_forward_f)();
extern SLAVE_FUN(batch_slave_use_backward_f)();
extern SLAVE_FUN(batch_slave_nouse_backward_f)();
void sw_batch_norm_use_forward_impl_f(
    float * bottom_data,
    float * top_data,
    float * blobs_0,
    float * blobs_1,
    float * blobs_2,
    float * mean_by_channel,
    float * variance_by_channel,
    float * temp_mutable,
    float * xnorm,
    float eps,
    int num,            //batch_size
    int channels,       //C
    int spatial_dim     //H*W
)
{
    int i,j,k;

#ifdef USE_NOBLAS
    for(i=0;i<num;++i)
    {
      for(j=0;j<channels;++j)
      {
        if(i==0)
        {
          if(blobs_2[0]==0)
          {
            mean_by_channel[j]=0.0;
            variance_by_channel[j]=0.0;
          }
          else
          {
            mean_by_channel[j]=blobs_0[j]/blobs_2[0];
            variance_by_channel[j]=blobs_1[j]/blobs_2[0];
          }
        }
        for(k=0;k<spatial_dim;++k)
        {
           temp_mutable[i*channels*spatial_dim+j*spatial_dim+k]=sqrt(variance_by_channel[j]+eps);
           top_data[i*channels*spatial_dim+j*spatial_dim+k]=(top_data[i*channels*spatial_dim+j*spatial_dim+k]-mean_by_channel[j])
                   / temp_mutable[i*channels*spatial_dim+j*spatial_dim+k];
        }
      }
    }


#endif

#ifdef USE_SWDNN

    BNData* param = (BNData*)malloc(sizeof(BNData));
    param->bottom_data=bottom_data;
    param->top_data=top_data;
    param->mean_by_channel=mean_by_channel;
    param->variance_by_channel=variance_by_channel;
    param->temp_mutable=temp_mutable;
    param->eps=eps;
    param->num=num;
    param->channels=channels;
    param->spatial_dim=spatial_dim;
    param->xnorm=xnorm;

    for(i=0;i<channels;++i)
    {
        if(blobs_2[0]==0)
        {
          mean_by_channel[i]=0.0;
          variance_by_channel[i]=0.0;
        }
        else
        {
          mean_by_channel[i]=blobs_0[i]/blobs_2[0];
          variance_by_channel[i]=blobs_1[i]/blobs_2[0];
        }
    }
    athread_spawn(batch_slave_use_forward_f,param);
    athread_join();
    free(param);
#endif
}

void sw_batch_norm_nouse_forward_impl_f(
    float * bottom_data,
    float * top_data,
    float * num_by_chans_,
    float * temp_mutable,
    float * blobs_0,
    float * blobs_1,
    float * blobs_2,
    float * moving_average_fraction_,
    float * xnorm,
    float eps,
    int num,
    int channels,
    int spatial_dim
)
{
    int i,j,k,m,base;
    float sum,bias_correction_factor;
#ifdef USE_NOBLAS
    float * mean = malloc(channels*sizeof(float));
    float * variance = malloc(channels*sizeof(float));
    for(i = 0; i < channels; ++i) {
      mean[i] = 0.;
      variance[i] = 0.0;
    }

    for(i=0;i<num;++i)
    {
      for(j=0;j<channels;++j)
      {
        for(k=0;k<spatial_dim;++k)
        {
          mean[j]+=bottom_data[i*channels*spatial_dim+j*spatial_dim+k];
        }
      }
    }

    for(i=0;i<channels;++i) mean[i]/=(float)(num*spatial_dim);

    for(i=0;i<num;++i)
    {
      for(j=0;j<channels;++j)
      {
        for(k=0;k<spatial_dim;++k)
        {
          top_data[i*channels*spatial_dim+j*spatial_dim+k]-=mean[j];
        }
      }
    }

    for(i=0;i<num;++i)
    {
      for(j=0;j<channels;++j)
      {
        for(k=0;k<spatial_dim;++k)
        {
          variance[j]+=(top_data[i*channels*spatial_dim+j*spatial_dim+k])*(top_data[i*channels*spatial_dim+j*spatial_dim+k]);
        }
      }
    }
    for(i=0;i<channels;++i) variance[i]/=(float)(num*spatial_dim);



    //compute and save moving average
    blobs_2[0] *= (*moving_average_fraction_);
    blobs_2[0] += 1;

    for(i=0;i<channels;++i)
    {
        blobs_0[i]=mean[i] + (*moving_average_fraction_)*blobs_0[i];
    }
    m = num / channels;
    bias_correction_factor = m > 1 ? ((float)(m))/((float)(m-1)) : 1;
    for(i=0;i<channels;++i)
    {
        blobs_1[i]=bias_correction_factor*variance[i]+(*moving_average_fraction_)*blobs_1[i];
    }


    for(i=0;i<num;++i)
    {
      for(j=0;j<channels;++j)
      {
        for(k=0;k<spatial_dim;++k)
        {
          temp_mutable[i*channels*spatial_dim+j*spatial_dim+k]=sqrt(variance[j]+eps);
          top_data[i*channels*spatial_dim+j*spatial_dim+k]/=temp_mutable[i*channels*spatial_dim+j*spatial_dim+k];
        }
      }
    }
    free(mean);
    free(variance);
    mean=NULL;
    variance=NULL;
#endif

#ifdef USE_SWDNN

    float * mean = malloc(channels*sizeof(float));
    float * variance = malloc(channels*sizeof(float));
    for(i=0;i<channels;++i)
    {
      mean[i]=0.0;
      variance[i]=0.0;
    }
    BNData* param = (BNData*)malloc(sizeof(BNData));
    param->bottom_data=bottom_data;
    param->top_data=top_data;
    param->mean_by_channel=mean;
    param->variance_by_channel=variance;
    param->temp_mutable=temp_mutable;
    param->num=num;
    param->channels=channels;
    param->spatial_dim=spatial_dim;
    param->eps=eps;
    param->xnorm=xnorm;

    athread_spawn(batch_slave_nouse_forward_f,param);
    athread_join();

    blobs_2[0] *= (*moving_average_fraction_);
    blobs_2[0] += 1;

    for(i=0;i<channels;++i)
    {
        blobs_0[i]=mean[i] + (*moving_average_fraction_)*blobs_0[i];
    }
    m = num / channels;
    bias_correction_factor = m > 1 ? ((float)(m))/((float)(m-1)) : 1;

    for(i=0;i<channels;++i)
    {
        blobs_1[i]=bias_correction_factor*variance[i]+(*moving_average_fraction_)*blobs_1[i];
    }

    free(param);
    free(mean);
    free(variance);
#endif

}

void sw_batch_norm_use_backward_impl_f(
    int temp_count,
    int num,
    int channels,
    int spatial_dim,
    const float * top_diff,
    const float * temp_cpu_data,
    float * bottom_diff
)
{
    int i,j,base;
#ifdef USE_NOBLAS
    //LOG(INFO)<<"enter backward float use";
    printf("enter backward float use\n");
    for(i=0;i<temp_count;++i)
    {
        bottom_diff[i]=top_diff[i]/temp_cpu_data[i];
    }
#endif

#ifdef USE_SWDNN
    BNData* param = (BNData*)malloc(sizeof(BNData));
    param->bottom_diff=bottom_diff;
    param->top_diff=top_diff;
    param->temp_mutable=temp_cpu_data;
    param->num=num;
    param->channels=channels;
    param->spatial_dim=spatial_dim;
    athread_spawn(batch_slave_use_backward_f,param);
    athread_join();
#endif
}

void sw_batch_norm_nouse_backward_impl_f(
    const float * top_data,
    const float * top_diff,
    float * bottom_diff,
    int num,
    int channels,
    int spatial_dim,
    int temp_count,
    const float * temp_cpu_data
)
{
    int i,j,k,base;
#ifdef USE_NOBLAS
    //LOG(INFO)<<"enter backward float nouse";
    printf("enter backward float nouse\n");
    float * alpha_by_channel = malloc(channels*sizeof(float));
    float * beta_by_channel = malloc(channels*sizeof(float));
    for(i=0;i<channels;++i)
    {
        alpha_by_channel[i]=0.0;
        beta_by_channel[i]=0.0;
    }

    for(i=0;i<num;++i)
    {
        for(j=0;j<channels;++j)
        {
            for(k=0;k<spatial_dim;++k)
            {
                alpha_by_channel[j]+=top_diff[i*channels*spatial_dim+j*spatial_dim+k];
                beta_by_channel[j]+=(top_diff[i*channels*spatial_dim+j*spatial_dim+k])*(top_data[i*channels*spatial_dim+j*spatial_dim+k]);
            }
        }
    }

    for(i=0;i<channels;++i)
    {
        alpha_by_channel[i]/=(float)(num*spatial_dim);
        beta_by_channel[i]/=(float)(num*spatial_dim);
    }


    for(i=0;i<num;++i)
    {
        for(j=0;j<channels;++j)
        {
            for(k=0;k<spatial_dim;++k)
            {
                bottom_diff[i*channels*spatial_dim+j*spatial_dim+k]=(top_diff[i*channels*spatial_dim+j*spatial_dim+k]
                -alpha_by_channel[j]
                -top_data[i*channels*spatial_dim+j*spatial_dim+k]*beta_by_channel[j])
                /(temp_cpu_data[i*channels*spatial_dim+j*spatial_dim+k]);
            }
        }
    }

    free(beta_by_channel);
    free(alpha_by_channel);
    beta_by_channel=NULL;
    alpha_by_channel=NULL;
#endif

#ifdef USE_SWDNN
    BNData* param = (BNData*)malloc(sizeof(BNData));
    param->bottom_diff=bottom_diff;
    param->top_diff=top_diff;
    param->top_data=top_data;
    param->temp_mutable=temp_cpu_data;
    param->num=num;
    param->channels=channels;
    param->spatial_dim=spatial_dim;
    athread_spawn(batch_slave_nouse_backward_f,param);
    athread_join();
#endif
}


// double is not finished
////////////////////////////////////////////////////////





void sw_batch_norm_use_forward_impl_d(
    double * bottom_data,
    double * top_data,
    double * blobs_0,
    double * blobs_1,
    double * blobs_2,
    double * mean_by_channel,
    double * variance_by_channel,
    double * temp_mutable,
    double eps,
    int num,            //batch_size
    int channels,       //C
    int spatial_dim     //H*W
)
{
    int i,j,base;
//to do
#ifdef ZMPE_TRANS
    for(i=0;i<channels;++i)
    {
        base=i*num*spatial_dim;
        mean_by_channel[i]=blobs_0[i]*blobs_2[0];
        variance_by_channel[i]=blobs_1[i]*blobs_2[0];
        for(j=base;j<base+num*spatial_dim;++j)
        {
            temp_mutable[j]=sqrt(variance_by_channel[i]+eps);
            top_data[j]=(top_data[j]-mean_by_channel[i])/temp_mutable[j];
            
        }
    }
#endif

#ifdef SW_TRANS

    // ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    // athread_spawn(batch_norm_forward_compute,param);
    // athread_join();

#endif
}

void sw_batch_norm_nouse_forward_impl_d(
    double * bottom_data,
    double * top_data,
    double * num_by_chans_,
    double * temp_mutable,
    double * blobs_0,
    double * blobs_1,
    double * blobs_2,
    double * moving_average_fraction_,
    double eps,
    int num,
    int channels,
    int spatial_dim
)
{
    int i,j,m,base;
    double sum,bias_correction_factor;
#ifdef ZMPE_TRANS
    double * mean = malloc(channels*sizeof(double));
    double * variance = malloc(channels*sizeof(double));
    for(i=0;i<channels;++i)
    {
        base=i*num*spatial_dim;
        sum=0.0;
        for(j=base;j<base+num*spatial_dim;++j)
        {
            sum += top_data[j];
        }
        sum /= (double)(num*spatial_dim);
        mean[i] = sum;

    }

    for(i=0;i<channels;++i)
    {
        base=i*num*spatial_dim;
        for(j=base;j<base+num*spatial_dim;++j)
        {
            top_data[j]-=mean[i];
        }
    }

    for(i=0;i<channels;++i)
    {
        base=i*num*spatial_dim;
        sum=0.0;
        for(j=base;j<base+num*spatial_dim;++j)
        {
            sum+=(top_data[j]*top_data[j]);
        }
        variance[i]=sum/(double)(num*spatial_dim);
    }

    //compute and save moving average
    blobs_2[0] *= (*moving_average_fraction_);
    blobs_2[0] += 1;

    for(i=0;i<channels;++i)
    {
        blobs_0[i]=mean[i] + (*moving_average_fraction_)*blobs_0[i];
    }
    m = num / channels;
    bias_correction_factor = m > 1 ? ((double)(m))/((double)(m-1)) : 1;
    for(i=0;i<channels;++i)
    {
        blobs_1[i]=bias_correction_factor*variance[i]+(*moving_average_fraction_)*blobs_1[i];
    }
    for(i=0;i<channels;++i)
    {
        base=i*num*spatial_dim;
        for(j=base;j<base+num*spatial_dim;++j)
        {
            temp_mutable[j]=sqrt(variance[i]+eps);
            top_data[j]=(top_data[j])/temp_mutable[j];
        }
    }
    printf("mymean[0]=%f\n",mean[0]);
    printf("myvar[0]=%f\n",variance[0]);
    free(mean);
    free(variance);
    mean=NULL;
    variance=NULL;
#endif

#ifdef SW_TRANS

#endif

}

void sw_batch_norm_use_backward_impl_d(
    int temp_count,
    const double * top_diff,
    const double * temp_cpu_data,
    double * bottom_diff
)
{
    int i,j,base;
#ifdef ZMPE_TRANS
   // LOG(INFO)<<"enter backward double use";
    printf("enter backward double use\n");
    for(i=0;i<temp_count;++i)
    {
        bottom_diff[i]=top_diff[i]/temp_cpu_data[i];
    }
#endif

#ifdef SW_TRANS

#endif
}

void sw_batch_norm_nouse_backward_impl_d(
    const double * top_data,
    const double * top_diff,
    double * bottom_diff,
    int num,
    int channels,
    int spatial_dim,
    int temp_count,
    const double * temp_cpu_data
)
{
    int i,j,base;
#ifdef ZMPE_TRANS
    //LOG(INFO)<<"enter backward double nouse";
    printf("enter backward double nouse\n");
    double * alpha_by_channel = malloc(channels*sizeof(double));
    double * beta_by_channel = malloc(channels*sizeof(double));
    double alpha;
    double beta;
    for(i=0;i<channels;++i)
    {
        alpha=0.0;
        beta=0.0;
        base=i*num*spatial_dim;
        for(j=base;j<base+num*spatial_dim;++j)
        {
            alpha+=top_diff[j];
            beta+=top_diff[j]*top_data[j];
        }
        alpha_by_channel[i]=alpha/(double)(num*spatial_dim);
        beta_by_channel[i]=beta/(double)(num*spatial_dim);
    }
    for(i=0;i<channels;++i)
    {
        base=i*num*spatial_dim;
        for(j=base;j<base+num*spatial_dim;++j)
        {
            bottom_diff[j]=(top_diff[j]
            -alpha_by_channel[i]
            -top_data[j]*beta_by_channel[i])
            /(temp_cpu_data[j]);
        }
    }


    free(beta_by_channel);
    free(alpha_by_channel);
    beta_by_channel=NULL;
    alpha_by_channel=NULL;
#endif

#ifdef SW_TRANS

#endif
}
