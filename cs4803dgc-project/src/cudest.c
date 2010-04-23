#include "cudest.h"
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/ioctl.h>

#define NVCTLDEV "/dev/nvidiactl"

// Reverse-engineered from strace and binary analysis.
typedef enum {
	NV_PREPARE_FIFO	= 0xc04846d2,
} nvioctls;

// FIXME we'll almost certainly need a rwlock protecting this
static int nvctl = -1;

typedef struct nvfifo { // FIXME just a placeholding guess
	uint64_t ob[9];	// 0x48 bytes
} nvfifo;

static CUresult
init_ctlfd(int fd){
	nvfifo fifodesc;

	memset(&fifodesc,0,sizeof(fifodesc));
	fifodesc.ob[2] = 0x35ull;
	fifodesc.ob[1] = 0x312e36332e353931ull;
	if(ioctl(fd,NV_PREPARE_FIFO,&fifodesc)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	return CUDA_SUCCESS;
}

CUresult cuInit(unsigned flags){
	CUresult r;
	int fd;

	if(flags){
		return CUDA_ERROR_INVALID_VALUE;
	}
	if((fd = open(NVCTLDEV,O_RDWR)) < 0){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if((r = init_ctlfd(fd)) != CUDA_SUCCESS){
		close(fd);
		return r;
	}
	if(nvctl >= 0){
		close(nvctl);
	}
	nvctl = fd;
	return CUDA_SUCCESS;
}
