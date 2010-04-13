#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BLOCK_SIZE 64

static int
dumpresults(const uint32_t *res,unsigned count){
	unsigned z,y;

	for(z = 0 ; z < count ; z += 8){
		for(y = 0 ; y < 8 ; ++y){
			if(printf("%9u ",res[z + y]) < 0){
				return -1;
			}
		}
		if(printf("\n") < 0){
			return -1;
		}
	}
	return 0;
}

__global__ void
memkernel(unsigned *aptr,unsigned *bptr,unsigned *results){
	__shared__ typeof(*results) psum[BLOCK_SIZE];
	unsigned z;

	psum[threadIdx.x] = results[threadIdx.x];
	while(aptr + threadIdx.x < bptr){
		//printf("%p %u\n",(unsigned *)(aptr + unit * threadIdx.x),*(unsigned *)(aptr + unit * threadIdx.x));
		if(aptr[threadIdx.x]){
			++psum[threadIdx.x];
		}
		aptr += BLOCK_SIZE;
	}
	results[threadIdx.x] = psum[threadIdx.x];
	__syncthreads();
}

__global__ void
cudasanity(uint32_t *res,unsigned byte){
	__shared__ uint32_t psum[BLOCK_SIZE];

	psum[threadIdx.x] = res[threadIdx.x];
	psum[threadIdx.x] = byte;
	res[threadIdx.x] = psum[threadIdx.x];
}

int main(void){
	unsigned hr[BLOCK_SIZE],*ptr;
	dim3 dblock(BLOCK_SIZE,1,1);
	dim3 dgrid(1,1);

	memset(hr,0,sizeof(hr));
	if(cudaMalloc(&ptr,sizeof(hr)) || cudaMemset(ptr,0x00,sizeof(hr))){
		return EXIT_FAILURE;
	}
//	cudasanity<<<dgrid,dblock>>>(ptr,0xf0);
	memkernel<<<dgrid,dblock>>>(ptr,(unsigned *)((char *)ptr + sizeof(hr)),ptr);
	if(cudaMemcpy(hr,ptr,sizeof(hr),cudaMemcpyDeviceToHost)){
		return EXIT_FAILURE;
	}
	if(cudaFree(ptr)){
		return EXIT_FAILURE;
	}
	if(dumpresults(hr,sizeof(hr) / sizeof(*hr))){
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
