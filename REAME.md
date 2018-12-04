# swDNN

#im2col

# I decide to use cmake to build this project
# why batch-im2col is wrong

# why batch-im2col is slow in some cases
It is not because of data movement in LDM.
bad DMA pattern?


# Add new layers
1. mkdir a directory using your new layer name in ./slave/
`mkdir ./slave/conv`
2. write your slave code with name sw_slave_###
vi sw_slave_conv.c
3. write your host code (interface to call slave code) in ./src/sw###.c
vi swconv.c
4. write your header file in ./include/sw###.h
vi swconv.h
5. write your test code in ./unitest/test_###.c
vi test_conv.c
6. update test header file in ./include/swtest.h
