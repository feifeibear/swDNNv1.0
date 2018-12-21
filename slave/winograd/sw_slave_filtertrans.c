#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "./include/swwinogradconv.h"

/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2018.Sep.19th
 *
 * winograd input transformation
 *
 * ************/
//(3, 3, Ni, No)
//divide among Ni*No
#define SIMDSIZE 4
void FJR_filter_trans(FilterData* param)
{
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int Ni = param->Ni;
  int No = param->No;
  //assert(Ni*No%64 == 0);
  int blkNum = 64;
  int blkSize = Ni*No/blkNum;
  while(blkSize > 650) {
    blkNum *= 2;
    blkSize = Ni*No/blkNum;
  }

  if(blkSize > 650) {
    if(0 == id)
      printf("FJR_filter_trans LDM usage overflow!\n");
  }
  //blkNum % 64 == 0

  float* local_input  = (float*) ldm_malloc(sizeof(float)*blkSize*9);
  int local_input_size = blkSize*9;

  float* local_output = (float*) ldm_malloc(sizeof(float)*blkSize*16);
  int local_output_size = blkSize*16;

  volatile int  input_replyget = 0, replyput = 0;
  dma_desc dma_get_input, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  dma_set_size(&dma_get_input, blkSize*9*sizeof(float));
  dma_set_bsize(&dma_get_input, blkSize*sizeof(float));
  dma_set_stepsize(&dma_get_input, (Ni*No - blkSize)*sizeof(float));

  dma_set_size(&dma_put_output, blkSize*16*sizeof(float));
  dma_set_bsize(&dma_put_output, blkSize*sizeof(float));
  dma_set_stepsize(&dma_put_output, (Ni*No - blkSize)*sizeof(float));

  int cBlk, cN;
  for(cBlk = id; cBlk < blkNum; cBlk += 64) {
    float* input_offset = (float*)param->filter + cBlk*blkSize;
    dma(dma_get_input, (long)(input_offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    //DMA get a (4,4,Ni) -> put(16,Ni)
    for(cN = 0; cN < blkSize; cN += 4) {
      floatv4 F[9];
      floatv4 c1[16];
      simd_load(F[0], local_input + 0*blkSize + cN);
      simd_load(F[1], local_input + 1*blkSize + cN);
      simd_load(F[2], local_input + 2*blkSize + cN);
      simd_load(F[3], local_input + 3*blkSize + cN);
      simd_load(F[4], local_input + 4*blkSize + cN);
      simd_load(F[5], local_input + 5*blkSize + cN);
      simd_load(F[6], local_input + 6*blkSize + cN);
      simd_load(F[7], local_input + 7*blkSize + cN);
      simd_load(F[8], local_input + 8*blkSize + cN);

      c1[0]  = F[0];
      c1[1]  = (F[0]+F[2]+F[1])*0.5f;
      c1[2]  = (F[0]+F[2]-F[1])*0.5f;
      c1[3]  = F[2];
      c1[4]  = (F[0]+F[6]+F[3])*0.5f;
      c1[5]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])+(F[1]+F[7]+F[4]))*0.25f; 
      c1[6]  = ((F[0]+F[6]+F[3])+(F[2]+F[8]+F[5])-(F[1]+F[7]+F[4]))*0.25f; 
      c1[7]  = (F[2]+F[8]+F[5])*0.5f;
      c1[8]  = (F[0]+F[6]-F[3])*0.5f; 
      c1[9]  = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])+(F[1]+F[7]-F[4]))*0.25f; 
      c1[10] = ((F[0]+F[6]-F[3])+(F[2]+F[8]-F[5])-(F[1]+F[7]-F[4]))*0.25f; 
      c1[11] = (F[2]+F[8]-F[5])*0.5f; 
      c1[12] = F[6]; 
      c1[13] = (F[6]+F[8]+F[7])*0.5f; 
      c1[14] = (F[6]+F[8]-F[7])*0.5f; 
      c1[15] = F[8]; 

      simd_store(c1[0],  local_output + 0*blkSize + cN);
      simd_store(c1[1],  local_output + 1*blkSize + cN);
      simd_store(c1[2],  local_output + 2*blkSize + cN);
      simd_store(c1[3],  local_output + 3*blkSize + cN);
      simd_store(c1[4],  local_output + 4*blkSize + cN);
      simd_store(c1[5],  local_output + 5*blkSize + cN);
      simd_store(c1[6],  local_output + 6*blkSize + cN);
      simd_store(c1[7],  local_output + 7*blkSize + cN);
      simd_store(c1[8],  local_output + 8*blkSize + cN);
      simd_store(c1[9],  local_output + 9*blkSize + cN);
      simd_store(c1[10], local_output + 10*blkSize + cN);
      simd_store(c1[11], local_output + 11*blkSize + cN);
      simd_store(c1[12], local_output + 12*blkSize + cN);
      simd_store(c1[13], local_output + 13*blkSize + cN);
      simd_store(c1[14], local_output + 14*blkSize + cN);
      simd_store(c1[15], local_output + 15*blkSize + cN);
    }

    dma(dma_put_output, (long)((float*)param->transFilter + cBlk*blkSize), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_input, sizeof(float)*local_input_size);
  ldm_free(local_output, sizeof(float)*local_output_size);

}//main func

