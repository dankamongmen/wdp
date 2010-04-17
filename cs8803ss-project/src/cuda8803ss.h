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

// Result codes. _CUDAFAIL means that the CUDA kernel raised an exception -- an
// expected mode of failure. _ERROR means some other exception occurred (abort
// the binary search of the memory).
typedef enum {
	CUDARANGER_EXIT_SUCCESS,
	CUDARANGER_EXIT_ERROR,
	CUDARANGER_EXIT_CUDAFAIL,
} cudadump_e;

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
		}else if(s != tmax && s != min){
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
	}while( (s = (tmax + min) / 2) );
	fprintf(stderr,"  All allocations failed.\n");
	return 0;
}

#ifdef __cplusplus
};
#endif

#endif
