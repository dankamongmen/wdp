#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include "cuda8803ss.h"

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
	unsigned oldptr = 0,ptr;
	uintmax_t total = 0,s;
	unsigned long zul;
	CUcontext ctx;
	int cerr;

	if(argc != 2){
		usage(*argv);
		exit(EXIT_FAILURE);
	}
	if(get_devno(argv[0],argv[1],&zul)){
		exit(EXIT_FAILURE);
	}
	if((cerr = init_cuda_ctx(zul,&ctx)) != CUDA_SUCCESS){
		fprintf(stderr,"Error initializing CUDA device %lu (%d)\n",zul,cerr);
		exit(EXIT_FAILURE);
	}
	if((s = cuda_alloc_max(stdout,&ptr,sizeof(unsigned))) == 0){
		fprintf(stderr,"Error allocating max on device %lu\n",zul);
		exit(EXIT_FAILURE);
	}
	zul = 0;
	do{
		if(printf("  Allocation at 0x%x (expected 0x%x)\n",ptr,oldptr) < 0){
			exit(EXIT_FAILURE);
		}
		total += s;
		if(ptr != oldptr){
			if(printf("  Memory hole: 0x%x->0x%x (%xb)\n",
				oldptr,ptr - 1,ptr - oldptr) < 0){
				exit(EXIT_SUCCESS);
			}
		}
		oldptr = ptr + s;
		++zul;
	}while( (s = cuda_alloc_max(stdout,&ptr,sizeof(unsigned))) );
	printf(" Got %ju (0x%jx) total bytes in %lu allocations.\n",total,total,zul);
	exit(EXIT_SUCCESS);
}
