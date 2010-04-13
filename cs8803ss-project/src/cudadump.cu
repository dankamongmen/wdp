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

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
static int
id_cuda(int dev,unsigned *mem,unsigned *tmem){
	struct cudaDeviceProp dprop;
	int major,minor,attr,cerr;
	void *str = NULL;
	CUcontext ctx;
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
	if((cerr = cuCtxCreate(&ctx,CU_CTX_BLOCKING_SYNC|CU_CTX_SCHED_YIELD,c)) != CUDA_SUCCESS){
		goto err;
	}
	if((cerr = cuMemGetInfo(mem,tmem)) != CUDA_SUCCESS){
		cuCtxDetach(ctx);
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

// Takes in start and end of memory area to be scanned, and fd. Returns the
// number of |unit|-byte words in this region, or 0 on error. mstart and mend
// must be |unit|-byte aligned, and mstart must be less than mend. Requires
// sufficient virtual memory to allocate the bitmap, and sufficient disk space
// for the backing file (FIXME we currently use a hole, so not quite...).
static uintmax_t
create_bitmap(uintptr_t mstart,uintptr_t mend,int fd,void **bmap,unsigned unit){
	int mflags;
	size_t s;

	if((mend - mstart) % 4096){
		mend = mstart + ((((mend - mstart) / 4096) + 1) * 4096);
	}
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
					unsigned gran,uint32_t *results){
	char min[40],max[40],dev[20];
	char * const argv[] = { RANGER, dev, min, max, NULL };
	pid_t pid;

	if((size_t)snprintf(min,sizeof(min),"%ju",off) >= sizeof(min) ||
			(size_t)snprintf(max,sizeof(max),"%ju",off + s) >= sizeof(max) ||
			(size_t)snprintf(dev,sizeof(dev),"%d",devno) >= sizeof(dev)){
		fprintf(stderr,"  Invalid arguments: %d %ju %ju\n",devno,min,max);
		return -1;
	}
	if((pid = fork()) < 0){
		fprintf(stderr,"  Couldn't fork (%s?)!\n",strerror(errno));
		return -1;
	}else if(pid == 0){
		if(execvp(RANGER,argv)){
			fprintf(stderr,"  Couldn't exec %s (%s?)!\n",RANGER,strerror(errno));
		}
		exit(CUDARANGER_EXIT_ERROR);
		//exit(dump_cuda(off,off + s,unit,results));
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
			// FIXME mark up the map
		}else if(WEXITSTATUS(status) == CUDARANGER_EXIT_CUDAFAIL){
			uintmax_t mid;

			mid = carve_range(off,off + s,gran);
			if(mid != s){
				if(divide_address_space(devno,off,mid,unit,gran,results)){
					return -1;
				}
				if(divide_address_space(devno,off + mid,s - mid,unit,gran,results)){
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
cudadump(int devno,uintmax_t tmem,int fd,unsigned unit,uintmax_t gran,uint32_t *results){
	void *map,*ptr;
	uintmax_t s;

	if((s = cuda_alloc_max(stdout,tmem / 2 + tmem / 3,&ptr,unit)) == 0){
		return -1;
	}
	printf("  Allocated %ju of %ju MB at %p:0x%jx\n",
			s / (1024 * 1024) + !!(s % (1024 * 1024)),
			tmem / (1024 * 1024) + !!(tmem % (1024 * 1024)),
			ptr,(uintmax_t)ptr + s);
	if(check_const_ram(CONSTWIN)){
		cudaFree(ptr);
		return -1;
	}
	// FIXME need to munmap(2) bitmap (especially on error paths!)
	if(create_bitmap(0,(uintptr_t)((char *)ptr + s),fd,&map,unit) == 0){
		fprintf(stderr,"  Error creating bitmap (%s?)\n",
				strerror(errno));
		cudaFree(ptr);
		return -1;
	}
	printf("  Verifying allocated region...\n");
	if(divide_address_space(devno,(uintmax_t)ptr,(s / gran) * gran,unit,gran,results)){
		fprintf(stderr,"  Sanity check failed!\n");
		cudaFree(ptr);
		return -1;
	}
	if(cudaFree(ptr)){
		fprintf(stderr,"  Error freeing CUDA memory (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
		return -1;
	}
	if(cudaThreadSynchronize()){
		fprintf(stderr,"  Sanity check failed! (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
		return -1;
	}
	printf("  Dumping %jub...\n",tmem - (uintptr_t)CONSTWIN);
	if(divide_address_space(devno,(uintptr_t)CONSTWIN,
				tmem - (uintptr_t)CONSTWIN,unit,gran,results)){
		fprintf(stderr,"  Error probing CUDA memory!\n");
		return -1;
	}
	if(check_const_ram(CONSTWIN)){
		return -1;
	}
	printf("  Dumping address space (%jub)...\n",(uintmax_t)0x100000000ull);
	if(divide_address_space(devno,0,0x100000000ull,unit,gran,results)){
		fprintf(stderr,"  Error probing CUDA memory!\n");
		return -1;
	}
	printf(" Success.\n");
	return 0;
}

int main(void){
	uintmax_t gran = 64 * 1024;	// Granularity of report / verification
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
		uint32_t hostresarr[BLOCK_SIZE];
		unsigned mem,tmem;
		uint32_t *resarr;
		int fd;

		printf(" %03d ",z);
		if(id_cuda(z,&mem,&tmem)){
			cudaError_t err;

			err = cudaGetLastError();
			fprintf(stderr," Error probing CUDA device %d (%s?)\n",
					z,cudaGetErrorString(err));
			return EXIT_FAILURE;
		}
		if((fd = open("localhost.dump",O_RDWR|O_CREAT,S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH)) < 0){
			fprintf(stderr," Error creating bitmap (%s?)\n",strerror(errno));
			return EXIT_FAILURE;
		}
		printf(" %03d ",z);
		if(id_cuda(z,&mem,&tmem)){
			cudaError_t err;

			err = cudaGetLastError();
			fprintf(stderr," Error probing CUDA device %d (%s?)\n",
					z,cudaGetErrorString(err));
			return EXIT_FAILURE;
		}
		if((fd = open("localhost.dump",O_RDWR|O_CREAT,S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH)) < 0){
			fprintf(stderr," Error creating bitmap (%s?)\n",strerror(errno));
			return EXIT_FAILURE;
		}
		if(cudaMalloc(&resarr,sizeof(hostresarr)) || cudaMemset(resarr,0,sizeof(hostresarr))){
			fprintf(stderr," Couldn't allocate result array (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
			return EXIT_FAILURE;
		}
		if(cudadump(z,tmem,fd,unit,gran,resarr)){
			close(fd);
			return EXIT_FAILURE;
		}
		if(cudaMemcpy(hostresarr,resarr,sizeof(hostresarr),cudaMemcpyDeviceToHost) || cudaFree(resarr)){
			fprintf(stderr," Couldn't free result array (%s?)\n",
				cudaGetErrorString(cudaGetLastError()));
			close(fd);
			return EXIT_FAILURE;
		}
		if(close(fd)){
			fprintf(stderr," Error closing bitmap (%s?)\n",strerror(errno));
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
