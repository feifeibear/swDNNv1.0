#ifndef SW_SOFTMAX_LAYER_IMPL_H_
#define SW_SOFTMAX_LAYER_IMPL_H_

typedef struct _tagSlaveSoftmaxParam
{
  int outer_num_,channels,inner_num_,dim;
  float * top_diff;
  float * top_data;
  float * bottom_diff;
  float * scale_data;
}SlaveSoftmaxParam;
void sw_softmax_forward_impl_f(
    const float* bottom_data,
    const float* sum_multiplier_,
    float* scale_data,
    float* top_data,
    int channels,
    int dim,
    int outer_num_,
    int inner_num_);

void sw_softmax_backward_impl_f(
    float * top_diff,
    float * top_data,
    float * bottom_diff,
    float * scale_data,
    int outer_num_,
    int channels,
    int inner_num_,
    int dim
    );
#endif
