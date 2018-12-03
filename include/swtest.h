#ifndef SWTEST_H_
#define SWTEST_H_
void test_col2im_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w);


void test_im2col_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w);

void test_im2col_zeropad_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w);

void test_im2col_zeropad_batch_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int batch_size);
#endif
