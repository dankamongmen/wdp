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
#define CONSTWIN ((unsigned *)0x10000u)

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

// Iterates over the specified memory region in units of CUDA's unsigned int.
// bptr must not be less than aptr. The provided array must be BLOCK_SIZE
// 32-bit integers; it holds the number of non-0 words seen by each of the
// BLOCK_SIZE threads.
__global__ void
readkernel(unsigned *aptr,const unsigned *bptr,uint32_t *results){
	__shared__ typeof(*results) psum[BLOCK_SIZE];

	psum[threadIdx.x] = results[threadIdx.x];
	while(aptr + threadIdx.x < bptr){
		++psum[threadIdx.x];
		if(aptr[threadIdx.x]){
			++psum[threadIdx.x];
		}
		aptr += BLOCK_SIZE;
	}
	results[threadIdx.x] = psum[threadIdx.x];
}

static cudadump_e
dump_cuda(uintmax_t tmin,uintmax_t tmax,unsigned unit,uint32_t *results){
	struct timeval time0,time1,timer;
	dim3 dblock(BLOCK_SIZE,1,1);
	int punit = 'M',cerr;
	dim3 dgrid(1,1,1);
	uintmax_t usec,s;
	float bw;

	if(cudaThreadSynchronize()){
		fprintf(stderr,"   Error prior to running kernel (%s)\n",
				cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	s = tmax - tmin;
	printf("   readkernel {%ux%u} x {%ux%ux%u} (0x%jx, 0x%jx (%jub), %u)\n",
		dgrid.x,dgrid.y,dblock.x,dblock.y,dblock.z,tmin,tmax,s,unit);
	gettimeofday(&time0,NULL);
	readkernel<<<dgrid,dblock>>>((unsigned *)tmin,(unsigned *)tmax,results);
	if( (cerr = cudaThreadSynchronize()) ){
		cudaError_t err;

		if(cerr != CUDA_ERROR_LAUNCH_FAILED && cerr != CUDA_ERROR_DEINITIALIZED){
			err = cudaGetLastError();
			fprintf(stderr,"   Error running kernel (%d, %s?)\n",
					cerr,cudaGetErrorString(err));
			return CUDARANGER_EXIT_ERROR;
		}
		//fprintf(stderr,"   Minor error running kernel (%d, %s?)\n",
				//cerr,cudaGetErrorString(cudaGetLastError()));
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
	printf("   elapsed time: %ju.%jus (%.3f %cB/s) res: %d\n",
			usec / 1000000,usec % 1000000,bw,punit,cerr);
	return CUDARANGER_EXIT_SUCCESS;
}

static int
dumpresults(const uint32_t *res,unsigned count){
	unsigned z,y;

	for(z = 0 ; z < count ; z += 8){
		for(y = 0 ; y < 8 ; ++y){
			if(printf("%9x ",res[z + y]) < 0){
				return -1;
			}
		}
		if(printf("\n") < 0){
			return -1;
		}
	}
	return 0;
}

// FIXME: we really ought take a bus specification rather than a device number,
// since the latter are unsafe across hardware removal/additions.
static void
usage(const char *a0){
	fprintf(stderr,"usage: %s devno addrmin addrmax\n",a0);
}

int main(int argc,char **argv){
	uint32_t hostres[BLOCK_SIZE],*resarr;
	unsigned long long min,max;
	unsigned unit = 4;		// Minimum alignment of references
	unsigned long zul;
	cudadump_e res;
	char *eptr;
	void *ptr;
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
		fprintf(stderr,"Error initializing CUDA device %d (%d, %s?)\n",
				zul,cerr,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cudaMalloc(&resarr,sizeof(hostres)) || cudaMemset(resarr,0x00,sizeof(hostres))){
		fprintf(stderr,"Error allocating %zu on device %d (%s?)\n",
			sizeof(hostres),zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cuda_alloc_max(NULL,0x100000000,&ptr,sizeof(unsigned)) == 0){
		fprintf(stderr,"Error allocating max on device %d (%s?)\n",
			zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if((res = dump_cuda(min,max,unit,resarr)) != CUDARANGER_EXIT_SUCCESS){
		return res;
	}
	if(cudaThreadSynchronize()){
		return res;
	}
	if(cudaMemcpy(hostres,resarr,sizeof(hostres),cudaMemcpyDeviceToHost)){
		fprintf(stderr,"Error copying %zu from device %d (%s?)\n",
			sizeof(hostres),zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(cudaFree(resarr)){
		fprintf(stderr,"Couldn't free %zu on device %d (%s?)\n",
			sizeof(hostres),zul,cudaGetErrorString(cudaGetLastError()));
		return CUDARANGER_EXIT_ERROR;
	}
	if(dumpresults(hostres,sizeof(hostres) / sizeof(*hostres))){
		return CUDARANGER_EXIT_ERROR;
	}
	return CUDARANGER_EXIT_SUCCESS;
}
