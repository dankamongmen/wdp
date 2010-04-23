#include <cuda.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

static void
usage(const char *argv){
	fprintf(stderr,"usage: %s\n",argv);
}

int main(int argc,char **argv){
	CUresult cerr;

	if(argc > 1){
		usage(*argv);
		exit(EXIT_FAILURE);
	}
	if( (cerr = cuInit(0)) ){
		fprintf(stderr,"Couldn't initialize CUDA (%d)\n",cerr);
		exit(EXIT_FAILURE);
	}
	printf("CUDA initialized.\n");
	exit(EXIT_SUCCESS);
}
