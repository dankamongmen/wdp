#ifndef CUDA8803SS_MAX
#define CUDA8803SS_MAX

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!

static uintmax_t
cuda_alloc_max(FILE *o,CUdeviceptr *ptr,unsigned unit){
	uintmax_t tmax = 1ul << ADDRESS_BITS;
	uintmax_t min = 0,s = tmax;

	if(o){ fprintf(o,"  Determining max allocation..."); }
	do{
		if(o) { fflush(o); }

		if(cuMemAlloc(ptr,s)){
			if((tmax = s) <= min + unit){
				tmax = min;
			}
		}else if(s != tmax){
			int cerr;

			if(o){ fprintf(o,"%jub...",s); }
			if((cerr = cuMemFree(*ptr)) ){
				fprintf(stderr,"  Couldn't free %jub at 0x%x (%d?)\n",
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
