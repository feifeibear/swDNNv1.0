cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
project (swDNN)
set (swim2coltest_VERSION_MAJOR 1)
set (swim2coltest_VERSION_MINOR 0)


set(CMAKE_C_COMPILER "sw5cc.new")
set(CMAKE_CXX_COMPILER "mpiCC")
set(CMAKE_AR "swar")
option(BUILD_TEST "Build the test sources" ON)

file(GLOB SLAVE_SOURCES "./slave/tensor_trans/*.c" "./slave/im2col/*.c"
  "./slave/col2im/*.c" "./slave/bias/*.c" "./slave/winograd/*.c"
  "./slave/implicit_conv/*.c" "./slave/batch_norm/*.c"
  "./slave/pooling/*.c" "./slave/softmax/*.c" "./slave/scale/*.c" "./slave/lstm/*.c")
file(GLOB HOST_SOURCES "./src/*.c")
file(GLOB TEST_SOURCES "./unitest/src/*.c" "./unitest/src/*.cpp")

add_definitions(-msimd -O2)
add_definitions(-D_MEM_ALIGN_)

set_source_files_properties(${SLAVE_SOURCES} COMPILE_FLAGS "-slave")
set_source_files_properties(${HOST_SOURCES} ${TEST_SOURCES} COMPILE_FLAGS "-host")
set_source_files_properties(${TEST_SOURCES} COMPILE_FLAGS "-host -std=c99")
#set_source_files_properties(${HOST_SOURCES} ${TEST_SOURCES} COMPILE_FLAGS
#  "-host -DUSE_SWDNN -DUSE_BLAS -DUSE_ALL")
add_library(swdnnlib ${SLAVE_SOURCES} ${HOST_SOURCES} ${TEST_SOURCES})

link_libraries("-hybrid -lm_slave -allshare")
link_libraries(swdnnlib)
link_libraries(rt)
link_libraries(stdc++)
link_directories(".")
include_directories(.)
include_directories(/home/export/online1/swyf/swdnn/fjr/2018-09/BLAS/swGEMM/)
set_source_files_properties(main.cpp COMPILE_FLAGS "-host")
add_executable(swdnn main.cpp)
target_link_libraries(swdnnlib
  /home/export/online1/swyf/swdnn/fjr/2018-09/BLAS/swGEMM/build/libswBLASlib.a)
target_link_libraries(swdnnlib /home/export/online1/swyf/swdnn/fjr/2018-09/BLAS/swGEMM/thirdparty/lib/cblas_LINUX0324.a)
target_link_libraries(swdnnlib /home/export/online1/swyf/swdnn/fjr/2018-09/BLAS/swGEMM/thirdparty/lib/libswblas0324.a)
#add_executable(samplesort test/samplesort.cpp)
enable_testing()
