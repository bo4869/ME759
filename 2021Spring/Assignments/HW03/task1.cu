#include <cuda.h>
#include <iostream>


__global__ void simpleKernel()
{
    int in = threadIdx.x + 1;
    int out = 1;
    for(int i = 1; i <=in; ++i) {
            out *= i;
        }

 std::printf("%d!=%d\n", in, out);
}
int main()
{
 const int numElems= 8;


 //invoke GPU kernel, with one block that has four threads
 simpleKernel<<<1,numElems>>>();
 cudaDeviceSynchronize();

 return 0;
}
