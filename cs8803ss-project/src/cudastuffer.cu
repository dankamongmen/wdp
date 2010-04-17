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
		usage(argv0);
		return -1;
	}
	return 0;
}

int main(int argc,char **argv){
	unsigned long zul;
	uintmax_t total,s;
	CUdeviceptr ptr;
	int cerr;

	if(argc != 2){
		usage(*argv);
		exit(EXIT_FAILURE);
	}
	if(get_devno(argv[0],argv[1],&zul)){
		exit(EXIT_FAILURE);
	}
	if((cerr = init_cuda(zul)) != CUDA_SUCCESS){
		fprintf(stderr,"Error initializing CUDA device %d (%d, %s?)\n",
				zul,cerr,cudaGetErrorString(cudaGetLastError()));
		exit(EXIT_FAILURE);
	}
	if((s = cuda_alloc_max(stdout,1ul << ADDRESS_BITS,&ptr,sizeof(unsigned))) == 0){
		fprintf(stderr,"Error allocating max on device %d (%s?)\n",
			zul,cudaGetErrorString(cudaGetLastError()));
		exit(EXIT_FAILURE);
	}
	total = s;
	while( (s = cuda_alloc_max(stdout,1ul << ADDRESS_BITS,&ptr,sizeof(unsigned))) ){
		total += s;
	}
	printf(" Got a total of %jub (0x%jx)\n",total,total);
	exit(EXIT_SUCCESS);
}
