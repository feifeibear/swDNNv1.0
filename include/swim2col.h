/***************************************
 * Created by Xin You
 * Date: 2017/8/8
 * Description: image to column functions
 *   accelerated in SW.
 **************************************/
#ifndef SWIM2COL_H_
#define SWIM2COL_H_
// data type: double
void swim2col_d(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    double* data_col);
// data type: float
void swim2col_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col);
// data type: float
void swim2col_zeropad_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col);



// data type: double
void swcol2im_d(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    double* data_im);

// data type: float
void swcol2im_f(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im);

typedef struct Im2colPara_st {
  void* data_im;
  void* data_col;
  int channels;
  int height;
  int width;
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int zeropad_col_rowsize;
  int zeropad_col_colsize;
  int batch_size;
}Im2colPara;


#endif
