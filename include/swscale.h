#ifndef SW_SCALE_LAYER_IMPL_H_
#define SW_SCALE_LAYER_IMPL_H_

extern void sw_scale_layer_d(const double* src,const double *scale, double* dst, const int outer_dim,const int inner_dim,const int scale_dim) ;
extern void sw_scale_layer_f(const float* src,const float *scale, float* dst, const int outer_dim,const int inner_dim,const int scale_dim) ;



#endif
