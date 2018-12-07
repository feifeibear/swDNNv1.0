#include <assert.h>
#include <athread.h>
#include "./include/swim2col.h"

#define LDM_MAX (64*1024)

extern void SLAVE_FUN(sw_im2col_large_stride_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_batch_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_zeropad_batch_trans_f)();
extern void SLAVE_FUN(sw_im2col_large_stride_d)();
extern void SLAVE_FUN(sw_im2col_large_stride_d)();
extern void SLAVE_FUN(sw_col2im_large_stride_f)();
extern void SLAVE_FUN(sw_im2col_large_d)();
extern void SLAVE_FUN(sw_im2col_large_f)();
extern void SLAVE_FUN(sw_col2im_large_d)();
extern void SLAVE_FUN(sw_col2im_large_f)();


// float version
// TODO
void swim2col_zeropad_batch_trans_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col, int batch_size) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  para->zeropad_col_colsize = (kernel_h * kernel_w * channels + 7)/8*8;
  para->zeropad_col_rowsize = (output_h * output_w + 127)/128*128;
  para->batch_size = batch_size;
  assert(dilation_h==1);
  assert(dilation_w==1);
  if(stride_h==1 && stride_w==1) {
    assert((width+2*pad_w)*sizeof(float)*batch_size<LDM_MAX);
    athread_spawn(sw_im2col_large_stride_zeropad_batch_trans_f,para);
    athread_join();
  } else {
    assert((width+2*pad_w + output_w)*sizeof(float)*batch_size<LDM_MAX);
    athread_spawn(sw_im2col_large_stride_zeropad_batch_trans_f,para);
    athread_join();
  }
  free(para);
}

// float version
void swim2col_zeropad_batch_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col, int batch_size) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  para->zeropad_col_colsize = (kernel_h * kernel_w * channels + 7)/8*8;
  para->zeropad_col_rowsize = (output_h * output_w + 127)/128*128;
  para->batch_size = batch_size;
  assert(dilation_h==1);
  assert(dilation_w==1);
  if(stride_h==1 && stride_w==1) {
    assert((width+2*pad_w)*sizeof(float)*batch_size<LDM_MAX);
    athread_spawn(sw_im2col_large_stride_zeropad_batch_f,para);
    athread_join();
  } else {
    assert((width+2*pad_w + output_w)*sizeof(float)*batch_size<LDM_MAX);
    athread_spawn(sw_im2col_large_stride_zeropad_batch_f,para);
    athread_join();
  }
  free(para);
}

// float version
void swim2col_zeropad_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  para->zeropad_col_rowsize = (output_w * output_h + 127)/128*128;
  para->zeropad_col_colsize = (kernel_h * kernel_w * channels + 7)/8*8;
  // check parameter Precondition of sw_im2col_large_d
  assert(dilation_h==1);
  assert(dilation_w==1);
  if(stride_h==1 && stride_w==1) {
    assert((width+2*pad_w)*sizeof(float)<LDM_MAX);
    // spawn
    //printf("SPAWN sw_im2col_large_f\n");
    //athread_spawn(sw_im2col_large_f,para);
    athread_spawn(sw_im2col_large_stride_zeropad_f,para);
    athread_join();
    //printf("sw_col2im_large_f end\n");
  } else {
    assert(((width+2*pad_w)+(width+2*pad_w-kernel_w)/stride_w+1)*sizeof(float)<LDM_MAX);
#ifdef PRINT_DEBUGINFO
    printf("SPAWN sw_im2col_large_stride_f\n");
#endif
    athread_spawn(sw_im2col_large_stride_zeropad_f,para);
    athread_join();
#ifdef PRINT_DEBUGINFO
    printf("sw_im2col_large_stride_f end\n");
#endif
  }

  free(para);
}


// float version
void swim2col_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  // check parameter Precondition of sw_im2col_large_d
  assert(dilation_h==1);
  assert(dilation_w==1);
  if(stride_h==1 && stride_w==1) {
    assert((width+2*pad_w)*sizeof(float)<LDM_MAX);
    athread_spawn(sw_im2col_large_f,para);
    athread_join();
  } else {
    assert(((width+2*pad_w)+(width+2*pad_w-kernel_w)/stride_w+1)*sizeof(float)<LDM_MAX);
    athread_spawn(sw_im2col_large_stride_f,para);
    athread_join();
  }

  free(para);
}

