#include <cuda.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include "cuda8803ss.h"

int main(void){
	CUresult cerr;

	cerr = cuInit(0);

	exit(cerr);
}
