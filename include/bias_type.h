#ifndef BIAS_TYPE_H_
#define BIAS_TYPE_H_





typedef struct _tagSlaveBiasParam
{
  int dim,bias_dim,inner_dim,outer_dim;
  float accum;
  float * bottom_data;
    float * top_data;
    float * bias_data;
    float * bias_diff;
    float * top_diff;
}SlaveBiasParam;


#endif
