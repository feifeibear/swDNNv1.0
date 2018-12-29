#include "include/swlstm.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define Dtype float
Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}
int test_lstm()
{
  int a[10],b[10],c[10];
  a[0]=64;a[1]=128;a[2]=256;a[3]=512;
  b[0]=128;b[1]=256;b[2]=512;b[3]=1024;b[4]=1600;
  int num,channels,w,h;
  int spatial_dim;
  int N,H,n,d;
  int i,j,k,z;
  int ii,jj,kk;
  int blob_size;
  float eps_=1e-4;
  struct timeval t1,t2;
  int outer_num_,inner_num_,dim;
  float sum,cont;

  float * clip_t=(float*)malloc(sizeof(float)*512);
  float * pre_gate_t=(float*)malloc(sizeof(float)*512*4*1600);
  float * h_to_gate=(float*)malloc(sizeof(float)*512*4*1600);
  float * c_t_1=(float*)malloc(sizeof(float)*512*1600);

  float * gate_t=(float*)malloc(sizeof(float)*512*4*1600);
  float * h_t=(float*)malloc(sizeof(float)*512*1600);
  float * c_t=(float*)malloc(sizeof(float)*512*1600);

  float * my_gate_t=(float*)malloc(sizeof(float)*512*4*1600);
  float * my_h_t=(float*)malloc(sizeof(float)*512*1600);
  float * my_c_t=(float*)malloc(sizeof(float)*512*1600);
  float * my_pre_gate_t=(float*)malloc(sizeof(float)*512*4*1600);

  float * backup_c_t_1 = c_t_1;
  float * backup_h_to_gate = h_to_gate;
  float * backup_my_h_t = my_h_t;
  float * backup_my_gate_t = my_gate_t;
  float * backup_my_c_t = my_c_t;



  char out[20]="0 0 time";

  printf("start\n");
  for(ii=0;ii<4;++ii){
    for(jj=0;jj<5;++jj){
        //if(k==1) break;
        printf("i=%d j=%d",ii,jj);
        N=a[ii];H=b[jj];

        blob_size=N*4*H;

        //with origin data
        //init
        for(z = 0; z < blob_size; ++z ) my_pre_gate_t[z]=pre_gate_t[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < blob_size; ++z ) h_to_gate[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < N; ++z) clip_t[z] = (rand()/(float)RAND_MAX)>0.5?1:0;
        for(z = 0; z < N*H; ++z) c_t_1[z] = rand()/(float)RAND_MAX;



        out[0]='0'+ii;
        out[2]='0'+jj;

        for (n = 0; n < N; ++n) {
          cont = clip_t[n];
          if (cont) {
            //caffe_add(4*H_, pre_gate_t, h_to_gate, pre_gate_t);
            for(d=0;d<4*H;++d)
            {
                my_pre_gate_t[d] += h_to_gate[d];
            }
          }
          for (d = 0; d < H; ++d) {
            // Apply nonlinearity
            my_gate_t[d] = sigmoid(my_pre_gate_t[d]);
            my_gate_t[H + d] = cont ? sigmoid(my_pre_gate_t[H + d]) : 0.0;
            my_gate_t[2*H + d] = sigmoid(my_pre_gate_t[2*H + d]);
            my_gate_t[3*H + d] = tanh(my_pre_gate_t[3*H + d]);

            // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
            my_c_t[d] = my_gate_t[H + d] * c_t_1[d] + my_gate_t[d] * my_gate_t[3*H + d];
            my_h_t[d] = my_gate_t[2*H + d] * tanh(my_c_t[d]);
          }
          my_h_t += H;
          my_c_t += H;
          c_t_1 += H;
          my_pre_gate_t += 4*H;
          my_gate_t += 4*H;
          h_to_gate += 4*H;
        }
        //to do


        c_t_1 = backup_c_t_1;
        h_to_gate = backup_h_to_gate;
        my_h_t = backup_my_h_t;
        my_gate_t = backup_my_gate_t;
        my_c_t = backup_my_c_t;


        double lstm_time=0;
        gettimeofday(&t1, NULL);
        sw_lstm_clip_forward_impl_f(
          (float*)clip_t,
          (float*)pre_gate_t,
          (float*)h_to_gate,
          (float*)gate_t,
          (float*)h_t,
          (float*)c_t_1,
          (float*)c_t,
          N,
          H
            );

        gettimeofday(&t2, NULL);
        lstm_time = TIME(t1,t2);
        double total_data_size=(N+19*N*H)*sizeof(float);
        int flag=1;
        for(i=0;i<N;++i)
        {
          for(j=0;j<H;++j)
          {
              if(h_t[i*H+j]-my_h_t[i*H+j]>1e-4
              || h_t[i*H+j]-my_h_t[i*H+j]<-1e-4)
              {
                flag=0;
                printf("lstm  error h_t %d %d = %f my_h_t %d %d = %f\n",i,j,h_t[i*H+j],i,j,my_h_t[i*H+j]);
                printf("gate t 0 %f gate t h %f gate t 2h %f gate t 3h %f\n",gate_t[i*4*H+j],gate_t[i*4*H+H+j],gate_t[i*4*H+2*H+j],gate_t[i*4*H+3*H+j]);
                printf("myte t 0 %f myte t h %f myte t 2h %f myte t 3h %f\n",my_gate_t[i*4*H+j],my_gate_t[i*4*H+H+j],my_gate_t[i*4*H+2*H+j],my_gate_t[i*4*H+3*H+j]);
                exit(0);
              }
          }
        }
        if(flag) printf("check ok\n");
        printf("lstm_layer %dx%d : Bandwidth : %f GB/s, time : %f sec\n", N, H, total_data_size/1e9/lstm_time, lstm_time);

    }
  }

  free(clip_t);
  free(pre_gate_t);
  free(h_to_gate);
  free(c_t_1);
  free(gate_t);
  free(h_t);
  free(c_t);
  free(my_gate_t);
  free(my_h_t);
  free(my_c_t);
  free(my_pre_gate_t);

  //athread_init();


  //athread_halt();

  return 0;
}

