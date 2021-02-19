#include <cuda.h>
#include <iostream>

__global__ void simpleKernel(int* data, int a)
{
 //this adds a value to a variable stored in global memory
 data[blockIdx.x*8+threadIdx.x] += blockIdx.x+ a*threadIdx.x;
}
int main()
{
 const int numElems= 8;
 int hA[numElems*2], *dA;
 
 //allocate memory on the device (GPU); zero out all entries in this device array 
 cudaMalloc((void**) &dA, sizeof(int) * numElems*2);
 cudaMemset(dA, 0, numElems*2* sizeof(int));

 //invoke GPU kernel, with one block that has four threads
 const int RANGE = 10;
int a = rand() % (RANGE + 1);
 simpleKernel<<<2,numElems>>>(dA,a);
 //bring the result back from the GPU into the hA
 cudaMemcpy(&hA, dA, sizeof(int) * numElems*2, cudaMemcpyDeviceToHost);
 //print out the result to confirm that things are looking good 
 std::cout << hA[0];
 for(int i = 1; i < numElems*2; i++)
  std::cout<< " "<<hA[i];
std::cout << std::endl;
 //release the memory allocated on the GPU 
 cudaFree(dA);
 return 0;
}