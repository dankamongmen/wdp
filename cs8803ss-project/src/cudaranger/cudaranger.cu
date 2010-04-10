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

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!
#define CONSTWIN ((unsigned *)0x10000u)
#define BLOCK_SIZE 512

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
static int
id_cuda(int dev,CUcontext *ctx){
	struct cudaDeviceProp dprop;
	int major,minor,attr,cerr;
	unsigned mem,tmem;
	void *str = NULL;
	CUdevice c;

	if((cerr = cuDeviceGet(&c,dev)) != CUDA_SUCCESS){
		return cerr;
	}
	if((cerr = cudaGetDeviceProperties(&dprop,dev)) != CUDA_SUCCESS){
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_WARP_SIZE,c);
	if(cerr != CUDA_SUCCESS || attr <= 0){
		return cerr;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,c);
	if(cerr != CUDA_SUCCESS || attr <= 0){
		return cerr;
	}
	if((cerr = cuDeviceComputeCapability(&major,&minor,c)) != CUDA_SUCCESS){
		return cerr;
	}
	if((str = malloc(CUDASTRLEN)) == NULL){
		return -1;
	}
	if((cerr = cuDeviceGetName((char *)str,CUDASTRLEN,c)) != CUDA_SUCCESS){
		goto err;
	}
	if((cerr = cuCtxCreate(ctx,CU_CTX_BLOCKING_SYNC|CU_CTX_SCHED_YIELD,c)) != CUDA_SUCCESS){
		goto err;
	}
	if((cerr = cuMemGetInfo(&mem,&tmem)) != CUDA_SUCCESS){
		cuCtxDetach(*ctx);
		goto err;
	}
	if(printf(" %d.%d %s %s %u/%uMB free %s\n",
		major,minor,
		dprop.integrated ? "Integrated" : "Standalone",(char *)str,
		mem / (1024 * 1024) + !!(mem / (1024 * 1024)),
		tmem / (1024 * 1024) + !!(tmem / (1024 * 1024)),
		dprop.computeMode == CU_COMPUTEMODE_EXCLUSIVE ? "(exclusive)" :
		dprop.computeMode == CU_COMPUTEMODE_PROHIBITED ? "(prohibited)" :
		dprop.computeMode == CU_COMPUTEMODE_DEFAULT ? "(shared)" :
		"(unknown compute mode)") < 0){
		cuCtxDetach(*ctx);
		cerr = -1;
		goto err;
	}
	free(str);
	return CUDA_SUCCESS;

err:	// cerr ought already be set!
	free(str);
	return cerr;
}

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
	printf("Compiled against CUDA version %d.%d. Linked against CUDA version %d.%d.\n",
			CUDAMAJMIN(CUDA_VERSION),CUDAMAJMIN(attr));
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
	printf("CUDA device count: %d\n",*count);
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
		return -1;
	}
	s = tmax - tmin;
	printf("   memkernel {%u x %u} x {%u x %u x %u} (0x%jx, 0x%jx (%jub), %u)\n",
		dgrid.x,dgrid.y,dblock.x,dblock.y,dblock.z,tmin,tmax,s,unit);
	gettimeofday(&time0,NULL);
	memkernel<<<dgrid,dblock>>>(tmin,tmax,unit);
	if( (cerr = cuCtxSynchronize()) ){
		fprintf(stderr,"   Error running kernel (%d?)\n",cerr);
		return -1;
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
	return 0;
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
	CUresult cerr;
	CUcontext ctx;
	char *eptr;

	if(argc != 4){
		usage(*argv);
		return EXIT_FAILURE;
	}
	if(((zul = strtoul(argv[1],&eptr,0)) == ULONG_MAX && errno == ERANGE)
			|| eptr == argv[1] || *eptr){
		fprintf(stderr,"Invalid device number: %s\n",argv[1]);
		usage(*argv);
		return EXIT_FAILURE;
	}
	if(((min = strtoull(argv[2],&eptr,0)) == ULLONG_MAX && errno == ERANGE)
			|| eptr == argv[2] || *eptr){
		fprintf(stderr,"Invalid minimum address: %s\n",argv[2]);
		usage(*argv);
		return EXIT_FAILURE;
	}
	if(((max = strtoull(argv[3],&eptr,0)) == ULLONG_MAX && errno == ERANGE)
			|| eptr == argv[3] || *eptr){
		fprintf(stderr,"Invalid maximum address: %s\n",argv[3]);
		usage(*argv);
		return EXIT_FAILURE;
	}
	if(init_cuda(&count)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	if(zul >= count){
		fprintf(stderr,"devno too large (%lu >= %d)\n",zul,count);
		usage(*argv);
		return EXIT_FAILURE;
	}
	if(id_cuda(zul,&ctx)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\nError probing CUDA device %lu (%s?)\n",
				zul,cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	if(dump_cuda(min,max,unit)){
		cuCtxDetach(ctx);
		return EXIT_FAILURE;
	}
	if((cerr = cuCtxDetach(ctx)) != CUDA_SUCCESS){
		fprintf(stderr,"\nError detaching context (%d?)\n",cerr);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
