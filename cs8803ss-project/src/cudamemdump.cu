#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
static int
id_cuda(int dev){
	struct cudaDeviceProp dprop;
	int major,minor,attr,cerr;
	unsigned mem;
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
	if((cerr = cuDeviceTotalMem(&mem,c)) != CUDA_SUCCESS){
		return cerr;
	}
	printf("%d.%d %s %s %uMB free %s\n",
		major,minor,
		dprop.integrated ? "Integrated" : "Standalone",(char *)str,
		mem / (1024 * 1024),
		dprop.computeMode == CU_COMPUTEMODE_EXCLUSIVE ? "(exclusive)" :
		dprop.computeMode == CU_COMPUTEMODE_PROHIBITED ? "(prohibited)" :
		dprop.computeMode == CU_COMPUTEMODE_DEFAULT ? "" :
		"(unknown compute mode)");
	free(str);
	return CUDA_SUCCESS;
}

#define CUDAMAJMIN(v) v / 1000, v % 1000

static int
init_cuda(void){
	int attr,count,z;
	int cerr;

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		/*if(cerr == CUDA_ERROR_NO_DEVICE){
			return 0;
		}*/
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
		if( (cerr = id_cuda(z)) ){
			return cerr;
		}
	}
	return CUDA_SUCCESS;
}

__global__ void memkernel(unsigned long *sum,unsigned long *words){
	const unsigned long *mem;
	int i;

	mem = sum;
	for(i = 0 ; i < 0x20000 ; ++i){
		*sum += *mem++;
		++*words;
	}
}

int main(void){
	unsigned long sum = 0,words = 0;
	void *ptr;

	if(init_cuda()){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	if(cudaMalloc(&ptr,sizeof(sum) * 2)){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error initializing CUDA (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	cudaMemset(ptr,0,sizeof(sum) * 2);
	memkernel<<<1,1>>>((typeof(&sum))ptr,(typeof(&sum))ptr + 1);
	cudaMemcpy(&sum,ptr,sizeof(sum),cudaMemcpyDeviceToHost);
	cudaMemcpy(&words,(typeof(&sum))ptr + 1,sizeof(sum),cudaMemcpyDeviceToHost);
	printf("sum: %u 0x%x\nwords: %u 0x%x (%u 0x%x bytes)\n",
			sum,sum,words,words,
			words * sizeof(sum),
			words * sizeof(sum));
	if(cudaFree(ptr) || words == 0){
		cudaError_t err;

		err = cudaGetLastError();
		fprintf(stderr,"Error dumping CUDA memory (%s?)\n",
				cudaGetErrorString(err));
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
