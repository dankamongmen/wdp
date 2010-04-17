#include <max.h>
#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>

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
	unsigned oldptr = 0,ptr;
	uintmax_t total = 0,s;
	unsigned long zul;
	int cerr;

	if(argc != 2){
		usage(*argv);
		exit(EXIT_FAILURE);
	}
	if(get_devno(argv[0],argv[1],&zul)){
		exit(EXIT_FAILURE);
	}
	if((cerr = init_cuda(zul)) != CUDA_SUCCESS){
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
