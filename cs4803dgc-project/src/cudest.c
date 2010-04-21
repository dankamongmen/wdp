#include "cudest.h"
#include <fcntl.h>
#include <unistd.h>

#define NVCTLDEV "/dev/nvidiactl"

// FIXME we'll almost certainly need a rwlock protecting this
static int nvctl = -1;

CUresult cuInit(unsigned flags){
	int fd;

	if(flags){
		return CUDA_ERROR_INVALID_VALUE;
	}
	if((fd = open(NVCTLDEV,O_RDWR)) < 0){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(nvctl >= 0){
		close(nvctl);
	}
	nvctl = fd;
	return CUDA_SUCCESS;
}
