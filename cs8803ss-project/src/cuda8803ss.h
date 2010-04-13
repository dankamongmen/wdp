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

#define BLOCK_SIZE 64

__device__ __constant__ unsigned constptr[1];

__global__ void constkernel(const unsigned *constmax){
	__shared__ unsigned psum[BLOCK_SIZE];
	unsigned *ptr;

	psum[threadIdx.x] = 0;
	// Accesses below 64k result in immediate termination, due to use of
	// the .global state space (2.0 provides unified addressing, which can
	// overcome this). That area's reserved for constant memory (.const
	// state space; see 5.1.3 of the PTX 2.0 Reference), from what I see.
	for(ptr = constptr ; ptr < constmax ; ptr += BLOCK_SIZE){
		psum[threadIdx.x] += ptr[threadIdx.x];
	}
}

#define CONSTWIN ((unsigned *)0x10000u)

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
memkernel(unsigned *aptr,const unsigned *bptr,uint32_t *results){
	__shared__ typeof(*results) psum[BLOCK_SIZE];

	results[threadIdx.x] = 0x44;
	psum[threadIdx.x] = results[threadIdx.x];
	while(aptr + threadIdx.x < bptr){
		++psum[threadIdx.x];
		/*if(aptr[threadIdx.x]){
			++psum[threadIdx.x];
		}*/
		aptr += BLOCK_SIZE;
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

	if(cudaThreadSynchronize()){
		fprintf(stderr,"   Error prior to running kernel (%s)\n",
				cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	s = tmax - tmin;
	printf("   memkernel {%ux%u} x {%ux%ux%u} (0x%jx, 0x%jx (%jub), %u)\n",
		dgrid.x,dgrid.y,dblock.x,dblock.y,dblock.z,tmin,tmax,s,unit);
	gettimeofday(&time0,NULL);
	memkernel<<<dgrid,dblock>>>((unsigned *)tmin,(unsigned *)tmax,results);
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

static int
check_const_ram(const unsigned *max){
	dim3 dblock(BLOCK_SIZE,1,1);
	dim3 dgrid(1,1,1);

	printf("  Verifying %jub constant memory...",(uintmax_t)max);
	fflush(stdout);
	constkernel<<<dblock,dgrid>>>(max);
	if(cuCtxSynchronize()){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error verifying constant CUDA memory (%s?)\n",
				cudaGetErrorString(err));
		return -1;
	}
	printf("good.\n");
	return 0;
}

static uintmax_t
cuda_alloc_max(FILE *o,uintmax_t tmax,void **ptr,unsigned unit){
	uintmax_t min = 0,s = tmax;

	if(o){ fprintf(o,"  Determining max allocation..."); }
	do{
		if(o) { fflush(o); }

		if(cudaMalloc(ptr,s)){
			if((tmax = s) <= min + unit){
				tmax = min;
			}
		}else if(s != tmax && s != min){
			if(o){ fprintf(o,"%jub...",s); }
			if(cudaFree(*ptr)){
				fprintf(stderr,"  Couldn't free %jub at %p (%s?)\n",
					s,*ptr,cudaGetErrorString(cudaGetLastError()));
				return 0;
			}
			min = s;
		}else{
			if(o) { fprintf(stderr,"%jub!\n",s); }
			return s;
		}
	}while( (s = (tmax + min) / 2) );
	fprintf(stderr,"  All allocations failed.\n");
	return 0;
}

#ifdef __cplusplus
};
#endif

#endif
