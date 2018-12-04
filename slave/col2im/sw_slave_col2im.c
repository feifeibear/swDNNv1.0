#include "slave.h"
#include "simd.h"
#include "dma.h"
#include "include/swim2col.h"

#define LDM_MAX (64*1024)
// Precondition: no stride and dilations. float data type
void sw_col2im_large_f(Im2colPara *para) {
  dma_desc dma_put_im, dma_get_col;
#define Type float
#define SIMDType floatv4
#define SIMDSIZE 4
  int pad_h = para->pad_h;
  int pad_w = para->pad_w;
  int height= para->height;
  int width = para->width;
  int kernel_h = para->kernel_h;
  int kernel_w = para->kernel_w;
  int output_h = height + 2 * pad_h - kernel_h + 1;
  int output_w = width + 2 * pad_w - kernel_w + 1;
  int channel_size = height*width;
  int out_channel_size = output_h*output_w*kernel_w*kernel_h;
  int id = athread_get_id(-1);
  // number of rows of <id> slave core.
  int local_row_size = (2*para->pad_h + para->height) * para->channels / 64
               + (id< ((2*para->pad_h + para->height) * para->channels % 64));
  // start row index of <id> slave core.
  int row_start= -para->pad_h+
                  id*((2*para->pad_h+para->height)*para->channels/64)
               + (id<((2*para->pad_h+para->height)*para->channels%64)?
                  id:((2*para->pad_h+para->height)*para->channels%64));
  int row_end = row_start+local_row_size; // row_start<= ir < row_end)
  // calculate the max size of rows calculated per iter.
  int max_batch_size = local_row_size;
  while(sizeof(Type)*max_batch_size*(output_w + width)>=LDM_MAX) {
    max_batch_size=max_batch_size/2; // half
  }
//#define __SINGLE_BATCH
#ifdef __SINGLE_BATCH
  max_batch_size=1;
#endif
  // no rows on duty
  if(max_batch_size<=0) return ;
  int batch_size; // batch_size<=max_batch_size
  // buffer size
  int local_buff_size= max_batch_size*output_w;
  int local_outbuff_size = max_batch_size*width;
  SIMDType* local_vbuffer = (SIMDType*)ldm_malloc(sizeof(Type)*local_buff_size);
  SIMDType* local_voutbuff= (SIMDType*)ldm_malloc(sizeof(Type)*local_outbuff_size);
  Type* local_buffer = (Type*)local_vbuffer;
  Type* local_outbuff= (Type*)local_voutbuff;
  // begin ptr of dma_get
  Type* local_buffer_begin;
  Type* input_ptr = (Type*)para->data_col;
  Type* output_ptr= (Type*)para->data_im;
  int input_row, ir, ic, channel, k;
  int output_row, output_col, outoff, inoff;
  volatile int input_replyget=0, replyput=0;
  // dma settings
  dma_set_op(&dma_get_col, DMA_GET);
  dma_set_mode(&dma_get_col, PE_MODE);
  dma_set_reply(&dma_get_col, &input_replyget);

  dma_set_op(&dma_put_im, DMA_PUT);
  dma_set_mode(&dma_put_im, PE_MODE);
  dma_set_reply(&dma_put_im, &replyput);

  //dma_set_size(&dma_get_col,output_w*sizeof(Type));
  //dma_set_size(&dma_put_im,width*sizeof(Type));

  // begin im2col
  int last_index, ik, ib, iout;
  ir = row_start;
  while(ir<row_end) {
    input_row = (ir+pad_h)%(height+2*pad_h)-pad_h;
    channel = (ir+pad_h)/(height+2*pad_h);
    outoff = channel*width*height;
    inoff= out_channel_size*channel;
    // the row is pad
    if(!((unsigned)input_row<(unsigned)height)) {
      // it is padding, skip
      ++ir;
      continue ;
    } else {
      if(input_row<kernel_h-1 || input_row>height-kernel_h) {
        batch_size=1;
      } else {
        last_index = channel*(height+2*pad_h)+height-kernel_h+1;
        batch_size = (last_index<row_end?last_index:row_end)-ir;
        if(batch_size>max_batch_size) batch_size=max_batch_size;
      }
      for(ic=0;ic<batch_size*width;++ic) {
        local_outbuff[ic] = 0.0;
      }
      // set DMA size by batch size
      dma_set_size(&dma_get_col,batch_size*output_w*sizeof(Type));
      // put data by dma
      for(ic=0;ic<kernel_w;++ic) {
        for(k=0;k<kernel_h;++k) {
          // put output_w size from local_buffer(ic) to 
          // output_ptr(channel,ic+k*kernel_w,(input_row-k+pad_h)*output_w)
          // output the (ic+k*kernel_w)-th data in each kernel
          output_row = ic+k*kernel_w;
          output_col = (input_row-k+pad_h)*output_w;
          if(output_col<0) break; // out of range
          if(output_col>=output_w*output_h) continue; // out of range
          dma( dma_get_col,
              (long)(input_ptr+output_row*(output_w*output_h)+output_col+inoff),
              (long)(local_buffer));
          dma_wait(&input_replyget, 1); input_replyget = 0;
          for(ib=0;ib<batch_size;++ib) {
            for(ik=0;ik<output_w;++ik) {
              iout = ik+ic-pad_w;
              if((unsigned)iout<(unsigned)width) {
                local_outbuff[ib*width+iout] += local_buffer[ib*output_w+ik];
              }
            }
          }
        }
      }
    }
    // get col data by dma
    dma_set_size(&dma_put_im,batch_size*width*sizeof(Type));
    dma(dma_put_im,(long)(output_ptr+input_row*width+outoff),(long)(local_outbuff));
    dma_wait(&replyput, 1); replyput = 0;
    ir += batch_size;
  }

  ldm_free(local_buffer,sizeof(Type)*local_buff_size);
  ldm_free(local_outbuff,sizeof(Type)*local_outbuff_size);
#undef Type
#undef SIMDType
#undef SIMDSIZE
}

// Precondition: no stride and dilations. double data type
void sw_col2im_large_d(Im2colPara *para) {
  dma_desc dma_put_im, dma_get_col;
#define Type double
#define SIMDType doublev4
#define SIMDSIZE 4
  int pad_h = para->pad_h;
  int pad_w = para->pad_w;
  int height= para->height;
  int width = para->width;
  int kernel_h = para->kernel_h;
  int kernel_w = para->kernel_w;
  int output_h = height + 2 * pad_h - kernel_h + 1;
  int output_w = width + 2 * pad_w - kernel_w + 1;
  int channel_size = height*width;
  int out_channel_size = output_h*output_w*kernel_w*kernel_h;
  int id = athread_get_id(-1);
  // number of rows of <id> slave core.
  int local_row_size = (2*para->pad_h + para->height) * para->channels / 64
               + (id< ((2*para->pad_h + para->height) * para->channels % 64));
  // start row index of <id> slave core.
  int row_start= -para->pad_h+
                  id*((2*para->pad_h+para->height)*para->channels/64)
               + (id<((2*para->pad_h+para->height)*para->channels%64)?
                  id:((2*para->pad_h+para->height)*para->channels%64));
  int row_end = row_start+local_row_size; // row_start<= ir < row_end)
  // calculate the max size of rows calculated per iter.
  int max_batch_size = local_row_size;
  while(sizeof(Type)*max_batch_size*(output_w + width)>=LDM_MAX) {
    max_batch_size=max_batch_size/2; // half
  }
//#define __SINGLE_BATCH
#ifdef __SINGLE_BATCH
  max_batch_size=1;
#endif
  // no rows on duty
  if(max_batch_size<=0) return ;
  int batch_size; // batch_size<=max_batch_size
  // buffer size
  int local_buff_size= max_batch_size*output_w;
  int local_outbuff_size = max_batch_size*width;
  SIMDType* local_vbuffer = (SIMDType*)ldm_malloc(sizeof(Type)*local_buff_size);
  SIMDType* local_voutbuff= (SIMDType*)ldm_malloc(sizeof(Type)*local_outbuff_size);
  Type* local_buffer = (Type*)local_vbuffer;
  Type* local_outbuff= (Type*)local_voutbuff;
  // begin ptr of dma_get
  Type* local_buffer_begin;
  Type* input_ptr = (Type*)para->data_col;
  Type* output_ptr= (Type*)para->data_im;
  int input_row, ir, ic, channel, k;
  int output_row, output_col, outoff, inoff;
  volatile int input_replyget=0, replyput=0;
  // dma settings
  dma_set_op(&dma_get_col, DMA_GET);
  dma_set_mode(&dma_get_col, PE_MODE);
  dma_set_reply(&dma_get_col, &input_replyget);

  dma_set_op(&dma_put_im, DMA_PUT);
  dma_set_mode(&dma_put_im, PE_MODE);
  dma_set_reply(&dma_put_im, &replyput);

  //dma_set_size(&dma_get_col,output_w*sizeof(Type));
  //dma_set_size(&dma_put_im,width*sizeof(Type));

  // begin im2col
  int last_index, ik, ib, iout;
  ir = row_start;
  while(ir<row_end) {
    input_row = (ir+pad_h)%(height+2*pad_h)-pad_h;
    channel = (ir+pad_h)/(height+2*pad_h);
    outoff = channel*width*height;
    inoff= out_channel_size*channel;
    // the row is pad
    if(!((unsigned)input_row<(unsigned)height)) {
      // it is padding, skip
      ++ir;
      continue ;
    } else {
      if(input_row<kernel_h-1 || input_row>height-kernel_h) {
        batch_size=1;
      } else {
        last_index = channel*(height+2*pad_h)+height-kernel_h+1;
        batch_size = (last_index<row_end?last_index:row_end)-ir;
        if(batch_size>max_batch_size) batch_size=max_batch_size;
      }
      for(ic=0;ic<batch_size*width;++ic) {
        local_outbuff[ic] = 0.0;
      }
      // set DMA size by batch size
      dma_set_size(&dma_get_col,batch_size*output_w*sizeof(Type));
      // put data by dma
      for(ic=0;ic<kernel_w;++ic) {
        for(k=0;k<kernel_h;++k) {
          // put output_w size from local_buffer(ic) to 
          // output_ptr(channel,ic+k*kernel_w,(input_row-k+pad_h)*output_w)
          // output the (ic+k*kernel_w)-th data in each kernel
          output_row = ic+k*kernel_w;
          output_col = (input_row-k+pad_h)*output_w;
          if(output_col<0) break; // out of range
          if(output_col>=output_w*output_h) continue; // out of range
          dma( dma_get_col,
              (long)(input_ptr+output_row*(output_w*output_h)+output_col+inoff),
              (long)(local_buffer));
          dma_wait(&input_replyget, 1); input_replyget = 0;
          for(ib=0;ib<batch_size;++ib) {
            for(ik=0;ik<output_w;++ik) {
              iout = ik+ic-pad_w;
              if((unsigned)iout<(unsigned)width) {
                local_outbuff[ib*width+iout] += local_buffer[ib*output_w+ik];
              }
            }
          }
        }
      }
    }
    // get col data by dma
    dma_set_size(&dma_put_im,batch_size*width*sizeof(Type));
    dma(dma_put_im,(long)(output_ptr+input_row*width+outoff),(long)(local_outbuff));
    dma_wait(&replyput, 1); replyput = 0;
    ir += batch_size;
  }

  ldm_free(local_buffer,sizeof(Type)*local_buff_size);
  ldm_free(local_outbuff,sizeof(Type)*local_outbuff_size);
#undef Type
#undef SIMDType
#undef SIMDSIZE
}

