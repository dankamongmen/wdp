#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda8803ss.h"

static void
usage(const char *a0){
	fprintf(stderr,"usage: %s devno\n",a0);
}

static int
basic_params(CUdeviceptr p,size_t s){
	CUdeviceptr p2;
	CUresult cerr;

	if( (cerr = cuMemAlloc(&p2,s)) || (cerr = cuMemsetD8(p2,0xff,s)) ){
		fprintf(stderr,"Couldn't alloc+init %zu base (%d)\n",s,cerr);
		return -1;
	}
	printf("Got secondary %zub allocation at %p\n",s,p2);
	if( (cerr = cuMemFree(p2)) ){
		fprintf(stderr,"Couldn't free %zu base (%d)\n",s,cerr);
		return -1;
	}
	// FIXME not very rigorous, not at all...[frown]
	printf("Minimum cuMalloc() alignment might be: %u\n",p2 - p);
	return 0;
}

#define BYTES_PER_KERNEL 4

__global__ void
touchbytes(CUdeviceptr ptr,uint32_t off,CUdeviceptr res){
	uint8_t b;

	b = *(unsigned char *)((uintptr_t)ptr + off + blockIdx.x);
	if(b == 0xff){
		*(uint32_t *)((uintptr_t)res + blockIdx.x * BYTES_PER_KERNEL) = 1;
	}
}

int main(int argc,char **argv){
	CUdeviceptr ptr,res;
	unsigned long zul;
	CUcontext ctx;
	CUresult cerr;
	size_t s,z;

	if(argc != 2 || getzul(argv[1],&zul)){
		usage(*argv);
		exit(EXIT_FAILURE);
	}
	if(init_cuda_ctx(zul,&ctx)){
		exit(EXIT_FAILURE);
	}
	s = sizeof(ptr);
	if( (cerr = cuMemAlloc(&ptr,s)) || (cerr = cuMemsetD8(ptr,0xff,s)) ){
		fprintf(stderr,"Couldn't alloc+init %zu base (%d)\n",s,cerr);
		exit(EXIT_FAILURE);
	}
	printf("Got base %zub allocation at %p\n",s,ptr);
	if(basic_params(ptr,s)){
		exit(EXIT_FAILURE);
	}
	if( (cerr = cuMemAlloc(&res,BYTES_PER_KERNEL * sizeof(uint32_t))) ||
			(cerr = cuMemsetD32(res,0,BYTES_PER_KERNEL)) ){
		fprintf(stderr,"Couldn't alloc+init %zu base (%d)\n",s,cerr);
		exit(EXIT_FAILURE);
	}
	if(res <= ptr){ // FIXME...see loop detect below
		fprintf(stderr,"Unexpected pointer arrangement (%p >= %p)\n",ptr,res);
		exit(EXIT_FAILURE);
	}
	printf("Got result %zub allocation at %p\n",BYTES_PER_KERNEL * sizeof(uint32_t),res);
	z = 0;
	while((cerr = cuCtxSynchronize()) == CUDA_SUCCESS){
		dim3 dg(1,1,1),db(BYTES_PER_KERNEL,1,1);

		printf("running on %zu\n",z);
		touchbytes<<<dg,db>>>(ptr,z,res);
		// FIXME check res
		if(((z += BYTES_PER_KERNEL) + ptr) > res){
			printf("Hit result array at %p; breaking loop\n",res);
			break;
		}
	}
	printf("Exited loop (ret: %d) at %zu\n",cerr,z);
	exit(EXIT_SUCCESS);
}
