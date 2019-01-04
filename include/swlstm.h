#ifndef LSTM_TYPE_H_
#define LSTM_TYPE_H_


typedef struct LSTMData_st {
        void * clip_t;
        void * pre_gate_t;
        void * h_to_gate;
        void * gate_t;
        void * h_t;
        void * c_t_1;
        void * c_t;
        int N_;
        int H_;
        int t;

}LSTMData;

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
);

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
);
/////////////
/*
typedef struct STDLSTMData_st{
        void * X;
        void * C_prev;
        void * cont;
        void * C;
        void * H;
        int num;
        int hidden_dim_;
}STDLSTMData;

void sw_std_lstm_forward_impl_f(
        float * X,
        float * C_prev,
        float * cont,
        float * C,
        float * H,
        int num,
        int hidden_dim_
);
*/

#endif
