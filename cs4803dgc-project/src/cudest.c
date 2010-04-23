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
} nvioctls;

// FIXME we'll almost certainly need a rwlock protecting this
static int nvctl = -1;

typedef struct nvhandshake {
	uint64_t ob[9];	// 0x48 bytes
} nvhandshake;

typedef uint32_t secondtype;

static secondtype result0xca;

typedef struct thirdtype {
	uint32_t ob[384];	// 1536 (0x600) bytes
} thirdtype;

static CUresult
init_ctlfd(int fd){
	nvhandshake hshake;
	thirdtype t3;

	memset(&hshake,0,sizeof(hshake));
	hshake.ob[2] = 0x35ull;
	hshake.ob[1] = 0x312e36332e353931ull;
	if(ioctl(fd,NV_HANDSHAKE,&hshake)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	{
		unsigned z;

		for(z = 0 ; z < sizeof(hshake.ob) / sizeof(*hshake.ob) ; ++z){
			printf("0x%2x 0x%jx\n",z * 8,(uintmax_t)hshake.ob[z]);
		}
	}
	if(ioctl(fd,NV_SECOND,&result0xca)){
		return CUDA_ERROR_INVALID_DEVICE;
	}
	memset(t3.ob,0,sizeof(t3.ob));
	t3.ob[0] = (uint32_t)-1;
	printf("%u\n",t3.ob[0]);
	if(ioctl(fd,NV_THIRD,t3)){
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
