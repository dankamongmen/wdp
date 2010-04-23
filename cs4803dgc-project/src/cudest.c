#include "cudest.h"
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/ioctl.h>

#define NVCTLDEV "/dev/nvidiactl"

// Reverse-engineered from strace and binary analysis.
typedef enum {
	NV_HANDSHAKE	= 0xc04846d2,
	NV_SECOND	= 0xc00446ca,
	NV_THIRD	= 0xc60046c8,
	NV_FOURTH	= 0xc00c4622,
} nvioctls;

// FIXME we'll almost certainly need a rwlock protecting this
static int nvctl = -1;

typedef struct nvhandshake {
	uint64_t ob[9];	// 0x48 bytes
} nvhandshake;

typedef uint32_t secondtype;

typedef struct thirdtype {
	uint32_t ob[384];	// 1536 (0x600) bytes
} thirdtype;

typedef struct fourthtype {
	uint32_t ob[3];		// 0xc (12) bytes
} fourthtype;

static thirdtype t3;
static fourthtype t4;
static secondtype result0xca;

static CUresult
init_ctlfd(int fd){
	nvhandshake hshake;

	memset(&hshake,0,sizeof(hshake));
	hshake.ob[2] = 0x35ull;
	hshake.ob[1] = 0x312e36332e353931ull;
	if(ioctl(fd,NV_HANDSHAKE,&hshake)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_SECOND,&result0xca)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	t3.ob[0] = (uint32_t)-1;
	if(ioctl(fd,NV_THIRD,t3)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_FOURTH,&t4)){
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
