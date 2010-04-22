#include <stdlib.h>
#include "cuda8803ss.h"

__global__ void quirkykernel(void){
	int __shared__ sharedvar;

	sharedvar = 0;
	__syncthreads();
	while(sharedvar != threadIdx.x);
		/***reconvergencepoint***/
	sharedvar++;
}

int main(void){
	if(init_cuda(0,NULL)){
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
