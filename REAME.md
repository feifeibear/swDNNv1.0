##swDNN

##Features
#im2col
1. support stride
2. support batch processing

#col2im


# TODO
# why batch-im2col is slow in some cases
It is not because of data movement in LDM.
bad DMA pattern?


## How to
#Use
mkdir ./build/
cd build && cmake .. && make

#Add new layers
1. mkdir a directory using your new layer name in ./slave/
`mkdir ./slave/conv`
write your slave code with name sw_slave_###
vi sw_slave_conv.c
2. write your host code (interface to call slave code) in ./src/sw###.c
vi swconv.c
3. write your header file in ./include/sw###.h
vi swconv.h

4. write your test code in ./unitest/src/test_###.c
6. update test header file in ./unitest/include/test_###.h

