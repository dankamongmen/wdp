#include <cuda.h>
#include <stdio.h>

int init_cuda(int devno,CUdevice *c){
	int attr,cerr;

	if((cerr = cuInit(0)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't initialize CUDA (%d), exiting.\n",cerr);
		return cerr;
	}
	if((cerr = cuDriverGetVersion(&attr)) != CUDA_SUCCESS){
		return cerr;
	}
	if(CUDA_VERSION > attr){
		fprintf(stderr,"Compiled against a newer version of CUDA than that installed, exiting.\n");
		return -1;
	}
	if((cerr = cuDeviceGet(c,devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get device reference (%d), exiting.\n",cerr);
		return cerr;
	}
	return CUDA_SUCCESS;
}

int init_cuda_ctx(int devno,CUcontext *cu){
	CUdevice c;
	int cerr;

	if((cerr = init_cuda(devno,&c)) != CUDA_SUCCESS){
		return cerr;
	}
	if((cerr = cuCtxCreate(cu,CU_CTX_BLOCKING_SYNC|CU_CTX_SCHED_YIELD,c)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't create context (%d), exiting.\n",cerr);
		return cerr;
	}
	return CUDA_SUCCESS;
}