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
init_cuda(unsigned *count){
	int attr,cerr,c;

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
	if((cerr = cuDeviceGetCount(&c)) != CUDA_SUCCESS){
		return cerr;
	}
	if(c <= 0){
		fprintf(stderr,"No CUDA devices found, exiting.\n");
		return -1;
	}
	*count = c;
	return CUDA_SUCCESS;
}

__global__ void
memkernel(uintptr_t aptr,const uintptr_t bptr,const unsigned unit){
	__shared__ unsigned psum[BLOCK_SIZE];

	psum[threadIdx.x] = 0;
	while(aptr + threadIdx.x * unit < bptr){
		psum[threadIdx.x] += *(unsigned *)(aptr + unit * threadIdx.x);
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

	if(tmin >= tmax){
		return CUDARANGER_EXIT_ERROR;
	}
	s = tmax - tmin;
	printf("   memkernel {%ux%u} x {%ux%ux%u} (0x%jx, 0x%jx (%jub), %u)\n",
		dgrid.x,dgrid.y,dblock.x,dblock.y,dblock.z,tmin,tmax,s,unit);
	gettimeofday(&time0,NULL);
	memkernel<<<dgrid,dblock>>>(tmin,tmax,unit);
	if( (cerr = cudaThreadSynchronize()) ){
		if(cerr != CUDA_ERROR_LAUNCH_FAILED){
			fprintf(stderr,"   Error running kernel (%d?)\n",cerr);
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
	unsigned count;
	char *eptr;

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
	if(init_cuda(&count)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA (%s?)\n",
				cudaGetErrorString(err));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cudaSetDevice(zul)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error selecting device %lu (%s?)\n",
				zul,cudaGetErrorString(err));
		if(zul > count){
			fprintf(stderr,"devno too large (%lu >= %d)\n",zul,count);
		}
		usage(*argv);
		return CUDARANGER_EXIT_ERROR;
	}
	// select device!
	return dump_cuda(min,max,unit);
}
