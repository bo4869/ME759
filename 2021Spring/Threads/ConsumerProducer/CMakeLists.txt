cmake_minimum_required (VERSION 3.13)

project (CTHREADS_GPUs LANGUAGES C CXX CUDA)                     

#set(CMAKE_CUDA_HOST_COMPILER $ENV{CU_CCBIN} CACHE PATH "Cuda host compiler dir")   
                                                                                   
set(CMAKE_BUILD_TYPE, Release CACHE STRING "Build type")                           
                                                                                   
add_executable(threadsConsProd 
main.cpp 
schedulingSupport.h
problemParams.h
consumer.cpp
consumer.h
producer.h
producer.cpp
)

target_compile_options(threadsConsProd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)


#add_subdirectory(kernelWrapper)