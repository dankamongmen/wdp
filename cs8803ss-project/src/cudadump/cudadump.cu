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

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!
#define CONSTWIN ((unsigned *)0x10000u)
#define BLOCK_SIZE 512

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
static int
id_cuda(int dev,unsigned *mem,unsigned *tmem,CUcontext *ctx){
	struct cudaDeviceProp dprop;
	int major,minor,attr,cerr;
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
	if((cerr = cuMemGetInfo(mem,tmem)) != CUDA_SUCCESS){
		cuCtxDetach(*ctx);
		goto err;
	}
	if(printf("%d.%d %s %s %u/%uMB free %s\n",
		major,minor,
		dprop.integrated ? "Integrated" : "Standalone",(char *)str,
		*mem / (1024 * 1024) + !!(*mem / (1024 * 1024)),
		*tmem / (1024 * 1024) + !!(*tmem / (1024 * 1024)),
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

// Takes in start and end of memory area to be scanned, and fd. Returns the
// number of |unit|-byte words in this region, or 0 on error. mstart and mend
// must be |unit|-byte aligned, and mstart must be less than mend. Requires
// sufficient virtual memory to allocate the bitmap, and sufficient disk space
// for the backing file (FIXME we currently use a hole, so not quite...).
static uintmax_t
create_bitmap(uintptr_t mstart,uintptr_t mend,int fd,void **bmap,unsigned unit){
	int mflags;
	size_t s;

	if(!unit || mstart % unit || mend % unit || mstart >= mend || fd < 0){
		errno = EINVAL;
		return 0;
	}
	mflags = MAP_SHARED;
#ifdef MAP_HUGETLB
	mflags |= MAP_HUGETLB;
#endif
	s = (mend - mstart) / unit / CHAR_BIT;
	*bmap = mmap(NULL,s,PROT_READ|PROT_WRITE,mflags,fd,0);
	if(*bmap == MAP_FAILED){
		return 0;
	}
	if(ftruncate(fd,s)){
		munmap(*bmap,s);
		return 0;
	}
	return s * CHAR_BIT;
}

static uintmax_t
cuda_alloc_max(uintmax_t tmax,CUdeviceptr *ptr,unsigned unit){
	uintmax_t min = 0,s;

	printf("  Determining max allocation...");
	while( (s = ((tmax + min) / 2) & (~(uintmax_t)0u << 2u)) ){
		fflush(stdout);

		if(cuMemAlloc(ptr,s)){
			if((tmax = s) <= min + unit){
				tmax = min;
			}
		}else if(s != tmax && s != min){
			printf("%jub...",s);
			// Arbitrary canary constant
			cuMemsetD32(*ptr,0x42069420,s / unit);
			if(cuMemFree(*ptr)){
				cudaError_t err;

				err = cudaGetLastError();
				fprintf(stderr,"  Couldn't free %jub (%s?)\n",
						s,cudaGetErrorString(err));
				return 0;
			}
			min = s;
		}else{
			printf("%jub!\n",s);
			return s;
		}
	}
	fprintf(stderr,"  All allocations failed.\n");
	return 0;
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
divide_address_space(int devno,uintmax_t off,uintmax_t s,unsigned unit,unsigned gran){
	pid_t pid;

	if((pid = fork()) < 0){
		fprintf(stderr,"  Couldn't fork (%s?)!\n",strerror(errno));
		return -1;
	}else if(pid == 0){
		char min[40],max[40],dev[20];
		char * const argv[] = { RANGER, dev, min, max, NULL };

		if((size_t)snprintf(min,sizeof(min),"%ju\n",off) >= sizeof(min) ||
				(size_t)snprintf(max,sizeof(max),"%ju\n",off + s) >= sizeof(max) ||
				(size_t)snprintf(dev,sizeof(dev),"%d\n",devno) >= sizeof(dev)){
			fprintf(stderr,"  Invalid arguments: %d %ju %ju\n",devno,min,max);
			exit(EXIT_FAILURE);
		}
		if(execvp(RANGER,argv)){
			fprintf(stderr,"  Couldn't exec %s (%s?)!\n",
					RANGER,strerror(errno));
		}
		exit(EXIT_FAILURE);
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
		if(!WIFEXITED(status) || WEXITSTATUS(status)){
			uintmax_t mid;

			mid = carve_range(off,off + s,gran);
			if(mid != s){
				if(divide_address_space(devno,off,mid,unit,gran)){
					return -1;
				}
				if(divide_address_space(devno,off + mid,s - mid,unit,gran)){
					return -1;
				}
			}
		}else{
			// FIXME Success! mark up the map
		}
	}
	return 0;
}

static int
dump_cuda(int devno,uintmax_t tmem,int fd,unsigned unit,uintmax_t gran){
	CUdeviceptr ptr;
	CUresult cerr;
	uintmax_t s;
	void *map;

	if((s = cuda_alloc_max(tmem,&ptr,unit)) == 0){
		return -1;
	}
	printf("  Allocated %ju of %ju MB at %p:0x%jx\n",
			s / (1024 * 1024) + !!(s % (1024 * 1024)),
			tmem / (1024 * 1024) + !!(tmem % (1024 * 1024)),
			ptr,(uintmax_t)ptr + s);
	if(check_const_ram(CONSTWIN)){
		return -1;
	}
	// FIXME need to set fd, free up bitmap (especially on error paths!)
	if(create_bitmap(0,(uintptr_t)((char *)ptr + s),fd,&map,unit) == 0){
		fprintf(stderr,"  Error creating bitmap (%s?)\n",
				strerror(errno));
		return -1;
	}
	printf("  Sanity checking allocated region...\n");
	if(divide_address_space(devno,(uintmax_t)ptr,(s / gran) * gran,unit,gran)){
		fprintf(stderr,"  Sanity check failed!\n");
		return -1;
	}
	if( (cerr = cuMemFree(ptr)) ){
		fprintf(stderr,"  Error freeing CUDA memory (%d?)\n",cerr);
		return -1;
	}
	if( (cerr = cuCtxSynchronize()) ){
		fprintf(stderr,"  Sanity check failed! (%d?)\n",cerr);
		return -1;
	}
	gran = tmem - (uintptr_t)CONSTWIN;
	printf("  Dumping %jub...\n",gran);
	if(divide_address_space(devno,(uintptr_t)CONSTWIN,gran,unit,gran)){
		fprintf(stderr,"  Error probing CUDA memory!\n");
		return -1;
	}
	return 0;
}

int main(void){
	uintmax_t gran = 1024 * 1024;	// Granularity of report / verification
	unsigned unit = 4;		// Minimum alignment of references
	int z,count;

	if(init_cuda(&count)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	for(z = 0 ; z < count ; ++z){
		unsigned mem,tmem;
		CUresult cerr;
		CUcontext ctx;
		int fd;

		printf(" %03d ",z);
		if(id_cuda(z,&mem,&tmem,&ctx)){
			cudaError_t err;

			err = cudaGetLastError();
			fprintf(stderr,"\nError probing CUDA device %d (%s?)\n",
					z,cudaGetErrorString(err));
			return EXIT_FAILURE;
		}
		if((fd = open("localhost.dump",O_RDWR|O_CREAT,S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH)) < 0){
			fprintf(stderr,"\nError creating bitmap (%s?)\n",strerror(errno));
			cuCtxDetach(ctx);
			return EXIT_FAILURE;
		}
		if(dump_cuda(z,tmem,fd,unit,gran)){
			close(fd);
			cuCtxDetach(ctx);
			return EXIT_FAILURE;
		}
		if(close(fd)){
			fprintf(stderr,"\nError closing bitmap (%s?)\n",strerror(errno));
			cuCtxDetach(ctx);
			return EXIT_FAILURE;
		}
		if((cerr = cuCtxDetach(ctx)) != CUDA_SUCCESS){
			fprintf(stderr,"\nError detaching context (%d?)\n",cerr);
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
