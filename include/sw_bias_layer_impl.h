#ifndef SW_BIAS_LAYER_IMPL_H_
#define SW_BIAS_LAYER_IMPL_H_



// typedef struct _tagSlaveBiasParam
// {
// 	int dim,bias_dim,inner_dim,outer_dim;
// 	float * bottom_data;
//     float * bias_data;
//     float * top_data;
// }SlaveBiasParam;

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


extern void sw_bias_impl_f(float * bottom_data,
                           float * top_data,
                           float * bias_data,
                           int dim,
                           int bias_dim,
                           int inner_dim,
                           int outer_dim);

extern void sw_bias_backward_impl_f(float * bias_diff,
                                    float * top_diff,
                                    float accum,
                                    int dim,
                                    int bias_dim,
                                    int inner_dim,
                                    int outer_dim);



#endif
