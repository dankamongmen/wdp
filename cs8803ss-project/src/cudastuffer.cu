#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "cuda8803ss.h"

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!

static int
init_cuda(int devno){
	int attr,cerr;
	CUcontext ctx;
	CUdevice c;

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		return cerr;
	}
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		return cerr;
	}
	if(CUDA_VERSION > attr){
		fprintf(stderr,"Compiled against a newer version of CUDA than that installed, exiting.\n");
		return -1;
	}
	if((cerr = cuDeviceGet(&c,devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get device reference, exiting.\n");
		return cerr;
	}
	if((cerr = cuCtxCreate(&ctx,0,c)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't create context, exiting.\n");
		return cerr;
	}
	return CUDA_SUCCESS;
}

static int
dumpresults(const uint32_t *res,unsigned count){
	unsigned z,y,nonzero;

	nonzero = 0;
	for(z = 0 ; z < count ; z += 8){
		for(y = 0 ; y < 8 ; ++y){
			if(printf("%9x ",res[z + y]) < 0){
				return -1;
			}
			if(res[z + y]){
				++nonzero;
			}
		}
		if(printf("\n") < 0){
			return -1;
		}
	}
	if(nonzero == 0){
		fprintf(stderr,"  All-zero results. Kernel probably didn't run.\n");
		return -1;
	}
	return 0;
}

// FIXME: we really ought take a bus specification rather than a device number,
// since the latter are unsafe across hardware removal/additions.
static void
usage(const char *a0){
	fprintf(stderr,"usage: %s devno\n",a0);
}

static int
get_devno(const char *argv0,const char *arg,unsigned long *zul){
	char *eptr;

	if(((*zul = strtoul(arg,&eptr,0)) == ULONG_MAX && errno == ERANGE)
			|| eptr == arg || *eptr){
		fprintf(stderr,"Invalid device number: %s\n",arg);
		printf("%lu %d\n",*zul,*eptr);
		usage(argv0);
		return -1;
	}
	return 0;
}

int main(int argc,char **argv){
	unsigned unit = 4;		// Minimum alignment of references
	unsigned long zul;
	CUdeviceptr ptr;
	cudadump_e res;
	int cerr;

	if(argc != 2){
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(get_devno(argv[0],argv[1],&zul)){
		return CUDARANGER_EXIT_ERROR;
	}
	if((cerr = init_cuda(zul)) != CUDA_SUCCESS){
		fprintf(stderr,"Error initializing CUDA device %d (%d, %s?)\n",
				zul,cerr,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cuda_alloc_max(NULL,1ul << ADDRESS_BITS,&ptr,sizeof(unsigned)) == 0){
		fprintf(stderr,"Error allocating max on device %d (%s?)\n",
			zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cuMemFree(ptr)){
		fprintf(stderr,"Warning: couldn't free memory\n");
	}
	return CUDARANGER_EXIT_SUCCESS;
}
