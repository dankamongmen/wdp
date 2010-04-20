#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "cuda8803ss.h"

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
static int
id_cuda(int dev,unsigned *mem,unsigned *tmem,int *state){
	struct cudaDeviceProp dprop;
	int major,minor,attr,cerr;
	void *str = NULL;
	CUcontext ctx;
	CUdevice c;

	*state = 0;
	if((cerr = cuDeviceGet(&c,dev)) != CUDA_SUCCESS){
		fprintf(stderr," Couldn't associative with device (%d)\n",cerr);
		return cerr;
	}
	if((cerr = cudaGetDeviceProperties(&dprop,dev)) != CUDA_SUCCESS){
		fprintf(stderr," Couldn't get device properties (%d)\n",cerr);
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
	if((cerr = cuCtxCreate(&ctx,CU_CTX_BLOCKING_SYNC|CU_CTX_SCHED_YIELD,c)) != CUDA_SUCCESS){
		fprintf(stderr," Couldn't create context (%d)\n",cerr);
		goto err;
	}
	if((cerr = cuMemGetInfo(mem,tmem)) != CUDA_SUCCESS){
		cuCtxDetach(ctx);
		goto err;
	}
	*state = dprop.computeMode;
	if(printf("%d.%d %s %s %u/%uMB free %s\n",
		major,minor,
		dprop.integrated ? "Integrated" : "Standalone",(char *)str,
		*mem / (1024 * 1024) + !!(*mem / (1024 * 1024)),
		*tmem / (1024 * 1024) + !!(*tmem / (1024 * 1024)),
		*state == CU_COMPUTEMODE_EXCLUSIVE ? "(exclusive)" :
		*state == CU_COMPUTEMODE_PROHIBITED ? "(prohibited)" :
		*state == CU_COMPUTEMODE_DEFAULT ? "(shared)" :
		"(unknown compute mode)") < 0){
		cerr = -1;
		goto err;
	}
	free(str);
	return CUDA_SUCCESS;

err:	// cerr ought already be set!
	free(str);
	return cerr;
}

__device__ __constant__ unsigned constptr[1];

__global__ void constkernel(const unsigned *constmax){
	__shared__ unsigned psum[BLOCK_SIZE];
	unsigned *ptr;

	psum[threadIdx.x] = 0;
	// Accesses below 64k result in immediate termination, due to use of
	// the .global state space (2.0 provides unified addressing, which can
	// overcome this). That area's reserved for constant memory (.const
	// state space; see 5.1.3 of the PTX 2.0 Reference), from what I see.
	for(ptr = constptr ; ptr < constmax ; ptr += BLOCK_SIZE){
		psum[threadIdx.x] += ptr[threadIdx.x];
	}
}

#define CONSTWIN ((unsigned *)0x10000u)

static int
check_const_ram(const unsigned *max){
	dim3 dblock(BLOCK_SIZE,1,1);
	dim3 dgrid(1,1,1);

	printf("  Verifying %jub constant memory...",(uintmax_t)max);
	fflush(stdout);
	constkernel<<<dblock,dgrid>>>(max);
	if(cuCtxSynchronize()){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"\n  Error verifying constant CUDA memory (%s?)\n",
				cudaGetErrorString(err));
		return -1;
	}
	printf("good.\n");
	return 0;
}

#define CUDAMAJMIN(v) v / 1000, v % 1000

static int
init_cuda(int *count){
	int attr,cerr;

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
	if((cerr = cuDeviceGetCount(count)) != CUDA_SUCCESS){
		return cerr;
	}
	if(*count <= 0){
		fprintf(stderr,"No CUDA devices found, exiting.\n");
		return -1;
	}
	printf("CUDA device count: %d\n",*count);
	return CUDA_SUCCESS;
}

// Returns the maxpoint of the first of at most two ranges into which the
// region will be divided, where a premium is placed on the first range being
// a multiple of gran.
static uintmax_t
carve_range(uintmax_t min,uintmax_t max,unsigned gran){
	uintmax_t mid;

	if(max < min){
		return 0;
	}
	// This way, we can't overflow given proper arguments. Simply averaging
	// min and max could overflow, resulting in an incorrect midpoint.
	mid  = min + (max - min) / 2;
	if((mid - min) % gran){
		if((mid += gran - ((mid - min) % gran)) > max){
			mid = max;
		}
	}
	return mid - min;
}

#define RANGER "out/cudaranger"

static int
divide_address_space(int devno,uintmax_t off,uintmax_t s,unsigned unit,
					unsigned gran,uint32_t *results,
					uintmax_t *worked){
	char min[40],max[40],dev[20];
	char * const argv[] = { RANGER, dev, min, max, NULL };
	pid_t pid;

	if((size_t)snprintf(min,sizeof(min),"0x%jx",off) >= sizeof(min) ||
			(size_t)snprintf(max,sizeof(max),"0x%jx",off + s) >= sizeof(max) ||
			(size_t)snprintf(dev,sizeof(dev),"%d",devno) >= sizeof(dev)){
		fprintf(stderr,"  Invalid arguments: %d 0x%jx 0x%jx\n",devno,off,off + s);
		return -1;
	}
	//printf("CALL: %s %s %s\n",dev,min,max);
	if((pid = fork()) < 0){
		fprintf(stderr,"  Couldn't fork (%s?)!\n",strerror(errno));
		return -1;
	}else if(pid == 0){
		if(execvp(RANGER,argv)){
			fprintf(stderr,"  Couldn't exec %s (%s?)!\n",RANGER,strerror(errno));
		}
		exit(CUDARANGER_EXIT_ERROR);
	}else{
		int status;
		pid_t w;

		while((w = wait(&status)) != pid){
			if(w < 0){
				fprintf(stderr,"  Error waiting (%s?)!\n",
						strerror(errno));
				return -1;
			}
		}
		if(!WIFEXITED(status) || WEXITSTATUS(status) == CUDARANGER_EXIT_ERROR){
			fprintf(stderr,"  Exception running %s %s %s %s\n",
					argv[0],argv[1],argv[2],argv[3]);
			return -1;
		}else if(WEXITSTATUS(status) == CUDARANGER_EXIT_SUCCESS){
			*worked += s;
		}else if(WEXITSTATUS(status) == CUDARANGER_EXIT_CUDAFAIL){
			uintmax_t mid;

			mid = carve_range(off,off + s,gran);
			if(mid != s){
				if(divide_address_space(devno,off,mid,unit,gran,results,worked)){
					return -1;
				}
				if(divide_address_space(devno,off + mid,s - mid,unit,gran,results,worked)){
					return -1;
				}
			}
		}else{
			fprintf(stderr,"  Unknown result code %d running"
				       " %s %s %s %s\n",WEXITSTATUS(status),
				       argv[0],argv[1],argv[2],argv[3]);
			return -1;
		}
	}
	return 0;
}

static int
cudadump(int devno,uintmax_t tmem,unsigned unit,uintmax_t gran,uint32_t *results){
	uintmax_t worked = 0,s;
	CUdeviceptr ptr;

	if(check_const_ram(CONSTWIN)){
		return -1;
	}
	if((s = cuda_alloc_max(stdout,&ptr,unit)) == 0){
		return -1;
	}
	printf("  Allocated %ju of %ju MB at 0x%jx:0x%jx\n",
			s / (1024 * 1024) + !!(s % (1024 * 1024)),
			tmem / (1024 * 1024) + !!(tmem % (1024 * 1024)),
			(uintmax_t)ptr,(uintmax_t)ptr + s);
	printf("  Verifying allocated region...\n");
	if(dump_cuda(ptr,ptr + (s / gran) * gran,unit,results)){
		cuMemFree(ptr);
		fprintf(stderr,"  Sanity check failed!\n");
		return -1;
	}
	if(cuMemFree(ptr)){
		fprintf(stderr,"  Error freeing CUDA memory (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
		return -1;
	}
	printf("  Dumping %jub...\n",tmem);
	if(divide_address_space(devno,0,tmem,unit,gran,results,&worked)){
		fprintf(stderr,"  Error probing CUDA memory!\n");
		return -1;
	}
	printf("  Readable: %jub/%jub (%f%%)\n",worked,tmem,(float)worked / tmem * 100);
	worked = 0;
	printf("  Dumping address space (%jub)...\n",(uintmax_t)0x100000000ull);
	if(divide_address_space(devno,0,0x100000000ull,unit,gran,results,&worked)){
		fprintf(stderr,"  Error probing CUDA memory!\n");
		return -1;
	}
	printf("  Readable: %jub/%jub (%f%%)\n",worked,0x100000000ull,(float)worked / 0x100000000 * 100);
	printf(" Success.\n");
	return 0;
}

#define GRAN_DEFAULT 4ul * 1024ul * 1024ul

static void
usage(const char *a0,int status){
	fprintf(stderr,"usage: %s [granularity]\n",a0);
	fprintf(stderr," default granularity: %lu\n",GRAN_DEFAULT);
	exit(status);
}

int main(int argc,char **argv){
	unsigned long gran;
	unsigned unit = 4;		// Minimum alignment of references
	int z,count;

	if(argc > 2){
		usage(argv[0],EXIT_FAILURE);
	}else if(argc == 1){
		if(getzul(argv[1],&gran)){
			usage(argv[0],EXIT_FAILURE);
		}
	}else{
		gran = GRAN_DEFAULT;
	}
	if(init_cuda(&count)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	for(z = 0 ; z < count ; ++z){
		uint32_t hostresarr[GRID_SIZE * BLOCK_SIZE];
		unsigned mem,tmem;
		uint32_t *resarr;
		int state;

		printf(" %03d ",z);
		if(id_cuda(z,&mem,&tmem,&state)){
			return EXIT_FAILURE;
		}
		if(state != CU_COMPUTEMODE_DEFAULT){
			printf("  Skipping device (put it in shared mode).\n",z);
			continue;
		}
		if(cudaMalloc(&resarr,sizeof(hostresarr)) || cudaMemset(resarr,0,sizeof(hostresarr))){
			fprintf(stderr," Couldn't allocate result array (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
			return EXIT_FAILURE;
		}
		if(cudadump(z,tmem,unit,gran,resarr)){
			return EXIT_FAILURE;
		}
		if(cudaMemcpy(hostresarr,resarr,sizeof(hostresarr),cudaMemcpyDeviceToHost) || cudaFree(resarr)){
			fprintf(stderr," Couldn't free result array (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
