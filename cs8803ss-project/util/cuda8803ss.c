#include <cuda.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <limits.h>

int init_cuda(int devno,CUdevice *c){
	int attr,cerr;

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't initialize CUDA (%d), exiting.\n",cerr);
		return cerr;
	}
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		return cerr;
	}
	if(CUDA_VERSION > attr){
		fprintf(stderr,"Compiled against a newer version of CUDA than that installed, exiting.\n");
		return -1;
	}
	if((cerr = cuDeviceGet(c,devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get device reference (%d), exiting.\n",cerr);
		return cerr;
	}
	return CUDA_SUCCESS;
}

int init_cuda_ctx(int devno,CUcontext *cu){
	CUdevice c;
	int cerr;

	if((cerr = init_cuda(devno,&c)) != CUDA_SUCCESS){
		return cerr;
	}
	if((cerr = cuCtxCreate(cu,CU_CTX_BLOCKING_SYNC|CU_CTX_SCHED_YIELD,c)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't create context (%d), exiting.\n",cerr);
		return cerr;
	}
	return CUDA_SUCCESS;
}

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!

uintmax_t cuda_alloc_max(FILE *o,CUdeviceptr *ptr,unsigned unit){
	uintmax_t tmax = 1ull << ADDRESS_BITS;
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

int getzul(const char *arg,unsigned long *zul){
	char *eptr;

	if(((*zul = strtoul(arg,&eptr,0)) == ULONG_MAX && errno == ERANGE)
			|| eptr == arg || *eptr){
		fprintf(stderr,"Expected an unsigned integer, got \"%s\"\n",arg);
		return -1;
	}
	return 0;
}
