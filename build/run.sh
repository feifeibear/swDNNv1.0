#bsub -b -I -q q_sw_expr -host_stack 1024 -n 1 -cgsp 64 -sw3run ../sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./swdnn im2col $1 $2 $3 $4 
#bsub -b -I -q q_sw_expr -host_stack 1024 -n 1 -cgsp 64 -sw3run ../sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./swdnn tensortrans $1 $2 $3 $4 
bsub -b -I -q q_sw_expr -host_stack 1024 -n 1 -cgsp 64 -sw3run /home/export/online1/swyf/swdnn/git/sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./swdnn winograd $1 $2 $3 $4 
