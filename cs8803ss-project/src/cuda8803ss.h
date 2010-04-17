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

#define GRID_SIZE 4
#define BLOCK_SIZE 128

// Result codes. _CUDAFAIL means that the CUDA kernel raised an exception -- an
// expected mode of failure. _ERROR means some other exception occurred (abort
// the binary search of the memory).
typedef enum {
	CUDARANGER_EXIT_SUCCESS,
	CUDARANGER_EXIT_ERROR,
	CUDARANGER_EXIT_CUDAFAIL,
} cudadump_e;

// Iterates over the specified memory region in units of CUDA's unsigned int.
// bptr must not be less than aptr. The provided array must be BLOCK_SIZE
// 32-bit integers; it holds the number of non-0 words seen by each of the
// BLOCK_SIZE threads.
__global__ void
readkernel(unsigned *aptr,const unsigned *bptr,uint32_t *results){
	__shared__ typeof(*results) psum[GRID_SIZE * BLOCK_SIZE];

	psum[BLOCK_SIZE * blockIdx.x + threadIdx.x] =
		results[BLOCK_SIZE * blockIdx.x + threadIdx.x];
	while(aptr + BLOCK_SIZE * blockIdx.x + threadIdx.x < bptr){
		++psum[BLOCK_SIZE * blockIdx.x + threadIdx.x];
		if(aptr[BLOCK_SIZE * blockIdx.x + threadIdx.x]){
			++psum[BLOCK_SIZE * blockIdx.x + threadIdx.x];
		}
		aptr += BLOCK_SIZE * GRID_SIZE;
	}
	results[BLOCK_SIZE * blockIdx.x + threadIdx.x] =
		psum[BLOCK_SIZE * blockIdx.x + threadIdx.x];
}

static cudadump_e
dump_cuda(uintmax_t tmin,uintmax_t tmax,unsigned unit,uint32_t *results){
	struct timeval time0,time1,timer;
	dim3 dblock(BLOCK_SIZE,1,1);
	int punit = 'M',cerr;
	dim3 dgrid(GRID_SIZE,1,1);
	uintmax_t usec,s;
	float bw;

	if(cudaThreadSynchronize()){
		fprintf(stderr,"   Error prior to running kernel (%s)\n",
				cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	s = tmax - tmin;
	printf("   readkernel {%ux%u} x {%ux%ux%u} (0x%08jx, 0x%08jx (0x%jxb), %u)\n",
		dgrid.x,dgrid.y,dblock.x,dblock.y,dblock.z,tmin,tmax,s,unit);
	gettimeofday(&time0,NULL);
	readkernel<<<dgrid,dblock>>>((unsigned *)tmin,(unsigned *)tmax,results);
	if( (cerr = cudaThreadSynchronize()) ){
		cudaError_t err;

		if(cerr != CUDA_ERROR_LAUNCH_FAILED && cerr != CUDA_ERROR_DEINITIALIZED){
			err = cudaGetLastError();
			fprintf(stderr,"   Error running kernel (%d, %s?)\n",
					cerr,cudaGetErrorString(err));
			return CUDARANGER_EXIT_ERROR;
		}
		//fprintf(stderr,"   Minor error running kernel (%d, %s?)\n",
				//cerr,cudaGetErrorString(cudaGetLastError()));
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

static uintmax_t
cuda_alloc_max(FILE *o,uintmax_t tmax,CUdeviceptr *ptr,unsigned unit){
	uintmax_t min = 0,s = tmax;

	if(o){ fprintf(o,"  Determining max allocation..."); }
	do{
		if(o) { fflush(o); }

		if(cuMemAlloc(ptr,s)){
			if((tmax = s) <= min + unit){
				tmax = min;
			}
		}else if(s != tmax && tmax - unit > min){
			int cerr;

			if(o){ fprintf(o,"%jub...",s); }
			if((cerr = cuMemFree(*ptr)) ){
				fprintf(stderr,"  Couldn't free %jub at %p (%d?)\n",
					s,*ptr,cerr);
				return 0;
			}
			min = s;
		}else{
			if(o) { fprintf(o,"%jub!\n",s); }
			return s;
		}
	}while( (s = ((tmax + min) / 2 / unit * unit)) );
	fprintf(stderr,"  All allocations failed.\n");
	return 0;
}

#ifdef __cplusplus
};
#endif

#endif
