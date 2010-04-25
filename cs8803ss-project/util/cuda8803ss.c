#include <cuda.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include <limits.h>

int init_cuda(int devno,CUdevice *c){
	int attr,cerr;
	CUdevice tmp;

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
	if(c == NULL){
		c = &tmp; // won't be passed pack, but allows device binding
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
	if((cerr = cuCtxCreate(cu,CU_CTX_SCHED_YIELD| CU_CTX_MAP_HOST,c)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't create context (%d), exiting.\n",cerr);
		return cerr;
	}
	return CUDA_SUCCESS;
}

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!

uintmax_t cuda_hostalloc_max(FILE *o,void **ptr,unsigned unit,unsigned flags){
	uintmax_t tmax = 1ull << ADDRESS_BITS;
	uintmax_t min = 0,s = tmax;

	if(o){ fprintf(o,"  Determining max allocation..."); }
	do{
		if(o) { fflush(o); }

		if(cuMemHostAlloc(ptr,s,flags)){
			if((tmax = s) <= min + unit){
				tmax = min;
			}
		}else if(s < tmax){
			int cerr;

			if(o){ fprintf(o,"%jub @ %p...",s,*ptr); }
			if((cerr = cuMemFreeHost(*ptr)) ){
				fprintf(stderr,"  Couldn't free %jub at %p (%d?)\n",
					s,*ptr,cerr);
				return 0;
			}
			min = s;
		}else{
			if(o) { fprintf(o,"%jub!\n",s); }
			return s;
		}
	}while( (s = ((tmax + min) * unit / 2 / unit)) );
	fprintf(stderr,"  All allocations failed.\n");
	return 0;
}

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
		}else if(s < tmax){
			int cerr;

			if(o){ fprintf(o,"%jub @ 0x%x...",s,*ptr); }
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
	}while( (s = ((tmax + min) * unit / 2 / unit)) );
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

#define CUDAMAJMIN(v) v / 1000, v % 1000

int init_cuda_alldevs(int *count){
	int attr,cerr;

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't initialize CUDA (%d)\n",cerr);
		return cerr;
	}
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get CUDA driver version (%d)\n",cerr);
		return cerr;
	}
	printf("Compiled against CUDA version %d.%d. Linked against CUDA version %d.%d.\n",
			CUDAMAJMIN(CUDA_VERSION),CUDAMAJMIN(attr));
	if(CUDA_VERSION > attr){
		fprintf(stderr,"Compiled against a newer version of CUDA than that installed, exiting.\n");
		return -1;
	}
	if((cerr = cuDeviceGetCount(count)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get CUDA device count (%d)\n",cerr);
		return cerr;
	}
	if(*count <= 0){
		fprintf(stderr,"No CUDA devices found, exiting.\n");
		return -1;
	}
	return CUDA_SUCCESS;
}
