#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>

#include "../include/swlstm.h"

extern SLAVE_FUN(lstm_slave_clip_forward_f)();
extern SLAVE_FUN(lstm_slave_noclip_forward_f)();
extern SLAVE_FUN(lstm_std_slave_forward_f)();
void sw_lstm_clip_forward_impl_f(
        float * clip_t,
        float * pre_gate_t,
        float * h_to_gate,
        float * gate_t,
        float * h_t,
        float * c_t_1,
        float * c_t,
        int N_,
        int H_
)
{
  LSTMData * param = (LSTMData*)malloc(sizeof(LSTMData));
  param->clip_t = clip_t;
  param->pre_gate_t = pre_gate_t;
  param->h_to_gate = h_to_gate;
  param->gate_t = gate_t;
  param->h_t = h_t;
  param->c_t_1 = c_t_1;
  param->c_t = c_t;
  param->N_ = N_;
  param->H_ = H_;
  athread_spawn(lstm_slave_clip_forward_f,param);
  athread_join();
  free(param);
}

void sw_lstm_noclip_forward_impl_f(
        int t,
        float * pre_gate_t,
        float * h_to_gate,
        float * gate_t,
        float * h_t,
        float * c_t_1,
        float * c_t,
        int N_,
        int H_
)
{
  LSTMData * param = (LSTMData*)malloc(sizeof(LSTMData));
  param->t = t;
  param->pre_gate_t = pre_gate_t;
  param->h_to_gate = h_to_gate;
  param->gate_t = gate_t;
  param->h_t = h_t;
  param->c_t_1 = c_t_1;
  param->c_t = c_t;
  param->N_ = N_;
  param->H_ = H_;
  athread_spawn(lstm_slave_clip_forward_f,param);
  athread_join();
  free(param);
}
/////////////////////
/*
void sw_std_lstm_forward_impl_f(
        float * X,
        float * C_prev,
        float * cont,
        float * C,
        float * H,
        int num,
        int hidden_dim_
)
{
    STDLSTMData * param = (STDLSTMData*)malloc(sizeof(STDLSTMData));
    param->X = X;
    param->C_prev = C_prev;
    param->cont = cont;
    param->C = C;
    param->H = H;
    param->num = num;
    param->hidden_dim_ = hidden_dim_;
    athread_spawn(lstm_std_slave_forward_f,param);
    athread_join();
    free(param);
}
*/
