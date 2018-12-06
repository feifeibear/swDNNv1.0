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
//(16, B, T, No) -> (B, Ro, Co, No)
#define SIMDSIZE 4
void FJR_output_trans(OutputData* param)
{
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int No = param->No;
  int B = param->B;
  int Ro = param->Ro;
  int Co = param->Co;
  int NR = Ro/2;
  int NC = Co/2;
  int T = NR*NC;

  float* local_input  = (float*) ldm_malloc(sizeof(float)*No*16);
  int local_input_size = No*16;

  float* local_output = (float*) ldm_malloc(sizeof(float)*No*4);
  int local_output_size = No*4;

  volatile int  input_replyget = 0, replyput = 0;
  dma_desc dma_get_input, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  dma_set_size(&dma_get_input, No*16*sizeof(float));
  dma_set_bsize(&dma_get_input, No*sizeof(float));
  dma_set_stepsize(&dma_get_input, (B*T-1)*No*sizeof(float));

  dma_set_size(&dma_put_output, No*4*sizeof(float));
  dma_set_bsize(&dma_put_output, No*2*sizeof(float));
  dma_set_stepsize(&dma_put_output, (Co-2)*No*sizeof(float));

  int cBlk, cNo;
  for(cBlk = id; cBlk < B*T; cBlk += 64) {
    int cB = cBlk/T;
    int cRo = cBlk%T/NC*2;
    int cCo = cBlk%NR*2;
    int cT = cBlk%T;
  //for(int cB = id; cB < B; cB += 64) {
  //  int cT = 0;
  //  for(int cRo = 0; cRo < Ro; cRo += 2)
  //    for(int cCo = 0; cCo < Co; cCo += 2) {
        dma(dma_get_input, (long)((float*)param->output + cB*No*T + cT*No), (long)(local_input));
        dma_wait(&input_replyget, 1); input_replyget = 0;

        for(cNo = 0; cNo < No; cNo += 4) {
          floatv4 c1[16];
          floatv4 c2[4];
          floatv4 tmp[16];
          simd_load(c1[0], local_input + 0*No + cNo);
          simd_load(c1[1], local_input + 1*No + cNo);
          simd_load(c1[2], local_input + 2*No + cNo);
          simd_load(c1[3], local_input + 3*No + cNo);
          simd_load(c1[4], local_input + 4*No + cNo);
          simd_load(c1[5], local_input + 5*No + cNo);
          simd_load(c1[6], local_input + 6*No + cNo);
          simd_load(c1[7], local_input + 7*No + cNo);
          simd_load(c1[8], local_input + 8*No + cNo);
          simd_load(c1[9], local_input + 9*No + cNo);
          simd_load(c1[10], local_input + 10*No + cNo);
          simd_load(c1[11], local_input + 11*No + cNo);
          simd_load(c1[12], local_input + 12*No + cNo);
          simd_load(c1[13], local_input + 13*No + cNo);
          simd_load(c1[14], local_input + 14*No + cNo);
          simd_load(c1[15], local_input + 15*No + cNo);

          // The tranformation manuamplified
          tmp[0] = c1[0]+c1[1]+ c1[2];
          tmp[1] = c1[1]-c1[2]- c1[3];
          tmp[2] = c1[4]+c1[5]+ c1[6];
          tmp[3] = c1[5]-c1[6]- c1[7];
          tmp[4] = c1[8]+c1[9]+ c1[10];
          tmp[5] = c1[9]-c1[10]- c1[11];
          tmp[6] = c1[12]+c1[13]+ c1[14];
          tmp[7] = c1[13]-c1[14]- c1[15];

          c2[0] = tmp[0]+tmp[2]+tmp[4];
          c2[1] = tmp[1]+tmp[3]+tmp[5];
          c2[2] = tmp[2]-tmp[4]-tmp[6];
          c2[3] = tmp[3]-tmp[5]-tmp[7];

          simd_store(c2[0], local_output + 0*No + cNo);
          simd_store(c2[1], local_output + 1*No + cNo);
          simd_store(c2[2], local_output + 2*No + cNo);
          simd_store(c2[3], local_output + 3*No + cNo);
        }

        dma(dma_put_output, (long)((float*)param->transOutput + cB*Ro*Co*No + (cRo*Co + cCo)*No), (long)(local_output));
        dma_wait(&replyput, 1); replyput = 0;
        cT++;
  }

  ldm_free(local_input, sizeof(float)*local_input_size);
  ldm_free(local_output, sizeof(float)*local_output_size);

}//main func

