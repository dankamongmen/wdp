#ifndef CUDA8803SS
#define CUDA8803SS

#ifdef __cplusplus
extern "C" {
#endif

	// this is some epic bullshit, done to work around issues in NVIDIA's
	// nvcc compiler...apologies all around

#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include "cuda8803ss.h"

#define BLOCK_SIZE 512

// Result codes. _CUDAFAIL means that the CUDA kernel raised an exception -- an
// expected mode of failure. _ERROR means some other exception occurred (abort
// the binary search of the memory).
typedef enum {
	CUDARANGER_EXIT_SUCCESS,
	CUDARANGER_EXIT_ERROR,
	CUDARANGER_EXIT_CUDAFAIL,
} cudadump_e;

// Iterates over the specified memory region in units of |unit| * BLOCK_SIZE
// bytes. bptr must not be less than aptr, and unit must not be 0. The provided
// array must be BLOCK_SIZE 32-bit integers; it holds the number of non-0 words
// seen by each of the BLOCK_SIZE threads.
__global__ void
memkernel(uintptr_t aptr,const uintptr_t bptr,const unsigned unit,
			uint32_t *results){
	__shared__ typeof(*results) psum[BLOCK_SIZE];

	psum[threadIdx.x] = results[threadIdx.x];
	while(aptr + threadIdx.x * unit < bptr){
		if(*(unsigned *)(aptr + unit * threadIdx.x)){
			++psum[threadIdx.x];
		}
		aptr += BLOCK_SIZE * unit;
	}
	results[threadIdx.x] = psum[threadIdx.x];
}

cudadump_e dump_cuda(uintmax_t tmin,uintmax_t tmax,unsigned unit,
						uint32_t *results){
	struct timeval time0,time1,timer;
	dim3 dblock(BLOCK_SIZE,1,1);
	int punit = 'M',cerr;
	dim3 dgrid(1,1,1);
	uintmax_t usec,s;
	float bw;

	if( (cerr = cuCtxSynchronize()) ){
		fprintf(stderr,"   Error prior to running kernel (%d).\n",cerr);
		return CUDARANGER_EXIT_ERROR;
	}
	s = tmax - tmin;
	printf("   memkernel {%ux%u} x {%ux%ux%u} (0x%jx, 0x%jx (%jub), %u)\n",
		dgrid.x,dgrid.y,dblock.x,dblock.y,dblock.z,tmin,tmax,s,unit);
	gettimeofday(&time0,NULL);
	memkernel<<<dgrid,dblock>>>(tmin,tmax,unit,results);
	if( (cerr = cuCtxSynchronize()) ){
		cudaError_t err;

		if(cerr != CUDA_ERROR_LAUNCH_FAILED && cerr != CUDA_ERROR_DEINITIALIZED){
			err = cudaGetLastError();
			fprintf(stderr,"   Error running kernel (%d, %s?)\n",
					cerr,cudaGetErrorString(err));
			return CUDARANGER_EXIT_ERROR;
		}
		return CUDARANGER_EXIT_CUDAFAIL;
	}
	gettimeofday(&time1,NULL);
	timersub(&time1,&time0,&timer);
	usec = (timer.tv_sec * 1000000 + timer.tv_usec);
	bw = (float)s / usec;
	if(bw > 1000.0f){
		bw /= 1000.0f;
		punit = 'G';
	}
	printf("   elapsed time: %ju.%jus (%.3f %cB/s) res: %d\n",
			usec / 1000000,usec % 1000000,bw,punit,cerr);
	return CUDARANGER_EXIT_SUCCESS;
}

#ifdef __cplusplus
};
#endif

#endif
