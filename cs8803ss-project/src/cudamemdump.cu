#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
static int
id_cuda(int dev,unsigned *mem){
	struct cudaDeviceProp dprop;
	int major,minor,attr,cerr;
	CUdevice c;
	void *str;

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
		free(str);
		return cerr;
	}
	if((cerr = cuDeviceTotalMem(mem,c)) != CUDA_SUCCESS){
		return cerr;
	}
	printf("%d.%d %s %s %uMB free %s\n",
		major,minor,
		dprop.integrated ? "Integrated" : "Standalone",(char *)str,
		*mem / (1024 * 1024),
		dprop.computeMode == CU_COMPUTEMODE_EXCLUSIVE ? "(exclusive)" :
		dprop.computeMode == CU_COMPUTEMODE_PROHIBITED ? "(prohibited)" :
		dprop.computeMode == CU_COMPUTEMODE_DEFAULT ? "" :
		"(unknown compute mode)");
	free(str);
	return CUDA_SUCCESS;
}

#define CUDAMAJMIN(v) v / 1000, v % 1000

static int
init_cuda(unsigned *mem){
	int attr,count,z;
	int cerr;

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
	if((cerr = cuDeviceGetCount(&count)) != CUDA_SUCCESS){
		return cerr;
	}
	if(count == 0){
		fprintf(stderr,"No CUDA devices found, exiting.\n");
		return -1;
	}
	printf("CUDA device count: %d\n",count);
	for(z = 0 ; z < count ; ++z){
		printf(" %03d ",z);
		if( (cerr = id_cuda(z,mem)) ){
			return cerr;
		}
	}
	return CUDA_SUCCESS;
}

#define ADDRESS_BITS 32u // FIXME 40 on compute capability 2.0!

#define BLOCK_SIZE 64 // FIXME bigger would likely be better

__global__ void memkernel(uintmax_t *sum,unsigned b){
	__shared__ typeof(*sum) psum;
	unsigned bp;

	psum = 0;
	for(bp = 0 ; bp < b ; bp += BLOCK_SIZE){
		psum += *(uintmax_t *)
			((uintmax_t)(sum + bp + threadIdx.x) % (1lu << ADDRESS_BITS));
	}
	sum[threadIdx.x] = psum;
}

int main(void){
	uintmax_t sums[BLOCK_SIZE],sum = 0;
	struct timeval time0,time1,timer;
	dim3 dblock(BLOCK_SIZE,1,1);
	unsigned mem;
	void *ptr;

	if(init_cuda(&mem)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
#define CHUNK (mem >> 2u)
	printf(" Want %ub (0x%x) of %ub (0x%x)\n",mem - CHUNK,mem - CHUNK,mem,mem);
	if(cudaMalloc(&ptr,mem - CHUNK)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	gettimeofday(&time0,NULL);
	memkernel<<<1,dblock>>>((typeof(&sum))ptr,(mem - CHUNK) / sizeof(*sums));
	if(cudaThreadSynchronize()){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error running kernel (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	gettimeofday(&time1,NULL);
	timersub(&time1,&time0,&timer);
	cudaMemcpy(sums,ptr,sizeof(sums),cudaMemcpyDeviceToHost);
	for(int i = 0 ; i < BLOCK_SIZE ; ++i){
		sum += sums[i];
	}
	printf(" sum: %ju 0x%jx\n",sum,sum);
	printf(" elapsed time: %luus (%.3f Mb/s)\n",
			timer.tv_sec * 1000000 + timer.tv_usec,
			(float)(mem - CHUNK) / (timer.tv_sec * 1000000 + timer.tv_usec));
	if(cudaFree(ptr) || cudaThreadSynchronize()){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error dumping CUDA memory (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
