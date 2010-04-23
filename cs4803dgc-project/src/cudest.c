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
	NV_FIFTH	= 0xc020462a,
	NV_I6		= 0xc048464d,
	NV_I7		= 0xc014462d,
	NV_I8		= 0xc0144632,
	NV_I9		= 0xc0204637,
	NV_IA		= 0xc020462b,
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

typedef struct type5 {
	uint32_t ob[8];		// 0x20 (32) bytes
} type5;

typedef struct type6 {
	uint32_t ob[12];	// 0x30 (48) bytes
} type6;

static type6 t6;
static type5 t5,ta;
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
	// FIXME ought be setting the rest of this up
	t5.ob[0] = t4.ob[0];
	t5.ob[1] = t4.ob[0];
	if(ioctl(fd,NV_FIFTH,&t5)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_I6,&t6)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_FIFTH,&t5)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_FIFTH,&t5)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_FIFTH,&t5)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_IA,&ta)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_I9,&ta)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_I9,&ta)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_FIFTH,&t5)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_FIFTH,&t5)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_FIFTH,&t5)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	if(ioctl(fd,NV_FIFTH,&t5)){
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

CUresult cuDeviceGet(CUdevice *d,int devno){
	if(devno < 0){
		return CUDA_ERROR_INVALID_VALUE;
	}
	d->devno = devno;
	return CUDA_SUCCESS;
}
