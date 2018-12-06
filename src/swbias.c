#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <simd.h>
//#include "caffe/swlayers/sw_bias_layer_impl.h"
//#include "caffe/swlayers/bias_type.h"
#include "include/swbias.h"

extern SLAVE_FUN(biasForward)();
extern SLAVE_FUN(biasBackward)();

void sw_bias_impl_f(float * bottom_data,
                    float * top_data,
                    float * bias_data,
                    int dim,
                    int bias_dim,
                    int inner_dim,
                    int outer_dim)                           
{

    SlaveBiasParam* param = (SlaveBiasParam*)malloc(sizeof(SlaveBiasParam));
    param->dim=dim;
    param->bias_dim=bias_dim;
    param->inner_dim=inner_dim;
    param->outer_dim=outer_dim;
    param->bottom_data=bottom_data;
    param->top_data=top_data;
    param->bias_data=bias_data;
    athread_spawn(biasForward,param);
   	athread_join();
    free(param);
}


void sw_bias_backward_impl_f(float * bias_diff,
                             float * top_diff,
                             float accum,
                             int dim,
                             int bias_dim,
                             int inner_dim,
                             int outer_dim)
{
    SlaveBiasParam* param = (SlaveBiasParam*)malloc(sizeof(SlaveBiasParam));
    param->accum=accum;
    param->dim=dim;
    param->bias_dim=bias_dim;
    param->inner_dim=inner_dim;
    param->outer_dim=outer_dim;
    param->top_diff=top_diff;
    param->bias_diff=bias_diff;
    athread_spawn(biasBackward,param);
    athread_join();
    free(param);
}
