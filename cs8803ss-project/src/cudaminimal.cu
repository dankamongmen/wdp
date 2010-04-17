#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

int main(void){
	CUresult cerr;

	if( (cerr = cuInit(0)) ){
		fprintf(stderr,"Couldn't initialize CUDA (%d)\n",cerr);
		exit(EXIT_FAILURE);
	}
	exit(EXIT_SUCCESS);
}
