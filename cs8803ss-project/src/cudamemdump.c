#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
static int
id_cuda(int dev){
	int major,minor,attr,cerr;
	unsigned mem;
	CUdevice c;
	char *str;

	if((cerr = cuDeviceGet(&c,dev)) != CUDA_SUCCESS){
		return -1;
	}
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_WARP_SIZE,c);
	if(cerr != CUDA_SUCCESS || attr <= 0){
		return -1;
	}
	// FIXME warp/cores is more our "SIMD width" than threads per core...
	cerr = cuDeviceGetAttribute(&attr,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,c);
	if(cerr != CUDA_SUCCESS || attr <= 0){
		return -1;
	}
	if((cerr = cuDeviceTotalMem(&mem,c)) != CUDA_SUCCESS){
		return -1;
	}
	// FIXME do something with mem -- new NUMA node?
	if((cerr = cuDeviceComputeCapability(&major,&minor,c)) != CUDA_SUCCESS){
		return -1;
	}
	if((str = malloc(CUDASTRLEN)) == NULL){
		return -1;
	}
	if((cerr = cuDeviceGetName(str,CUDASTRLEN,c)) != CUDA_SUCCESS){
		free(str);
		return -1;
	}
	printf("%s\n",str);
	return 0;
}

#define CUDAMAJMIN(v) v / 1000, v % 1000

static int
init_cuda(void){
	int attr,count,z;
	CUresult cerr;

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		/*if(cerr == CUDA_ERROR_NO_DEVICE){
			return 0;
		}*/
		return cerr;
	}
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		return cerr;
	}
	printf("Compiled against CUDA version %d.%d\nLinked against CUDA version %d.%d.\n",
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
	return 0;
}

int main(void){
	int err;

	if( (err = init_cuda()) ){
		if(err > 0){
			fprintf(stderr,"Error initializing CUDA (%s?)\n",
					cudaGetErrorString(err));
		}
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
