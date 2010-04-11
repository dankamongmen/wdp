#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "cuda8803ss.h"

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!
#define CONSTWIN ((unsigned *)0x10000u)
#define BLOCK_SIZE 512

#define CUDAMAJMIN(v) v / 1000, v % 1000

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

__global__ void
memkernel(uintptr_t aptr,const uintptr_t bptr,const unsigned unit){
	__shared__ unsigned psum[BLOCK_SIZE];

	psum[threadIdx.x] = 0;
	psum[threadIdx.x] += *(unsigned *)(aptr + unit * threadIdx.x);
	while(aptr + threadIdx.x * unit < bptr){
		//psum[threadIdx.x] += *(unsigned *)(aptr + unit * threadIdx.x);
		aptr += BLOCK_SIZE * unit;
	}
}

static int
dump_cuda(uintmax_t tmin,uintmax_t tmax,unsigned unit){
	struct timeval time0,time1,timer;
	dim3 dblock(BLOCK_SIZE,1,1);
	int punit = 'M',cerr;
	dim3 dgrid(1,1,1);
	uintmax_t usec,s;
	float bw;

	s = tmax - tmin;
	printf("   memkernel {%ux%u} x {%ux%ux%u} (0x%jx, 0x%jx (%jub), %u)\n",
		dgrid.x,dgrid.y,dblock.x,dblock.y,dblock.z,tmin,tmax,s,unit);
	gettimeofday(&time0,NULL);
	memkernel<<<dgrid,dblock>>>(tmin,tmax,unit);
	if( (cerr = cudaThreadSynchronize()) ){
		cudaError_t err;

		if(cerr != CUDA_ERROR_LAUNCH_FAILED){
			err = cudaGetLastError();
			fprintf(stderr,"   Error running kernel (%d, %s?)\n",
					cerr,cudaGetErrorString(err));
			return CUDARANGER_EXIT_ERROR;
		}
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
	printf("   elapsed time: %ju.%jus (%.3f %cB/s)\n",
			usec / 1000000,usec % 1000000,bw,punit);
	return CUDARANGER_EXIT_SUCCESS;
}

// FIXME: we really ought take a bus specification rather than a device number,
// since the latter are unsafe across hardware removal/additions.
static void
usage(const char *a0){
	fprintf(stderr,"usage: %s devno addrmin addrmax\n",a0);
}

int main(int argc,char **argv){
	unsigned long long min,max;
	unsigned unit = 4;		// Minimum alignment of references
	unsigned long zul;
	char *eptr;
	int cerr;

	if(argc != 4){
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(((zul = strtoul(argv[1],&eptr,0)) == ULONG_MAX && errno == ERANGE)
			|| eptr == argv[1] || *eptr){
		fprintf(stderr,"Invalid device number: %s\n",argv[1]);
		printf("%lu %d\n",zul,*eptr);
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(((min = strtoull(argv[2],&eptr,0)) == ULLONG_MAX && errno == ERANGE)
			|| eptr == argv[2] || *eptr){
		fprintf(stderr,"Invalid minimum address: %s\n",argv[2]);
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(((max = strtoull(argv[3],&eptr,0)) == ULLONG_MAX && errno == ERANGE)
			|| eptr == argv[3] || *eptr){
		fprintf(stderr,"Invalid maximum address: %s\n",argv[3]);
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if(max <= min){
		fprintf(stderr,"Invalid arguments: max (%ju) <= min (%ju)\n",
				max,min);
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	if((cerr = init_cuda(zul)) != CUDA_SUCCESS){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA device %d (%d, %s?)\n",
				zul,cerr,cudaGetErrorString(err));
		return CUDARANGER_EXIT_ERROR;
	}
	return dump_cuda(min,max,unit);
}
