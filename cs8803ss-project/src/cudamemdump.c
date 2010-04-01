#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

// CUDA must already have been initialized before calling cudaid().
#define CUDASTRLEN 80
#define CORES_PER_NVPROCESSOR 8 //  taken from CUDA 2.3 SDK's deviceQuery.cpp
static int
init_cuda(void){
	int attr,count;
	CUresult cerr;
	/*
	CUdevice c;
	char *str;
	*/

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		/*if(cerr == CUDA_ERROR_NO_DEVICE){
			return 0;
		}*/
		return cerr;
	}
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		return cerr;
	}
	printf("Compiled against CUDA version %d\nLinked against CUDA version %d\n",
			CUDA_VERSION,attr);
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
