#include <stdio.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/types.h>

void *mmap64(void *addr,size_t len,int prot,int flags,int fd,off_t off){
	static void *(*shim_mmap)(void *,size_t,int,int,int,off_t);
	void *r;

	if(shim_mmap == NULL){
		const char *msg;

		fprintf(stderr,"shimming system's mmap(2)\n");
		if((shim_mmap = dlsym(RTLD_NEXT,"mmap")) == NULL){
			fprintf(stderr,"got a NULL mmap(2)\n");
			errno = EPERM;
			return MAP_FAILED;
		}
		if( (msg = dlerror()) ){
			fprintf(stderr,"couldn't shim mmap(2): %s\n",msg);
			errno = EPERM;
			return MAP_FAILED;
		}
	}
	printf("mmap fd %d\n",fd);
	r = shim_mmap(addr,len,prot,flags,fd,off);
	printf("mmap result: %p\n",r);
	return r;
}

int ioctl(int fd,int req,uintptr_t op){//,unsigned o1,unsigned o2){
	static int (*shim_ioctl)(int,int,uintptr_t,int,int);
	static int (*shim_ioctl3)(int,int,uintptr_t);
	const uint32_t *dat = (const uint32_t *)op;
	int r,s,z;

	if(shim_ioctl == NULL){
		const char *msg;

		fprintf(stderr,"shimming system's ioctl(2)\n");
		if((shim_ioctl = dlsym(RTLD_NEXT,"ioctl")) == NULL){
			fprintf(stderr,"got a NULL ioctl(2)\n");
			errno = EPERM;
			return -1;
		}
		if( (msg = dlerror()) ){
			fprintf(stderr,"couldn't shim ioctl(2): %s\n",msg);
			errno = EPERM;
			return -1;
		}
		if((shim_ioctl3 = dlsym(RTLD_NEXT,"ioctl")) == NULL){
			fprintf(stderr,"got a NULL ioctl(2)\n");
			errno = EPERM;
			return -1;
		}
		if( (msg = dlerror()) ){
			fprintf(stderr,"couldn't shim ioctl(2): %s\n",msg);
			errno = EPERM;
			return -1;
		}
	}
	s = (req >> 16u) & 0x3fff;
	printf("ioctl %x, %d-byte param, fd %d\t",req & 0xff,s,fd);
	for(z = 0 ; z < s ; z += 4){
		printf("\x1b[1m");
		if(z % 16 == 0 && z){
			printf("0x%04x\t\t\t\t",z);
		}
		if(dat[z / 4]){
			printf("\x1b[32m");
		}
		printf("0x%08x ",dat[z / 4]);
		printf("\x1b[0m");
		if(z % 16 == 12){
			printf("\n");
		}
	}
	if(z % 16){
		printf("\n");
	}
	r = shim_ioctl3(fd,req,op);
	printf("\x1b[1m\x1b[34mRESULT: %d\x1b[0m\t\t\t",r);
	if(r == 0){
		for(z = 0 ; z < s ; z += 4){
			printf("\x1b[1m");
			if(z % 16 == 0 && z){
				printf("0x%04x\t\t\t\t",z);
			}
			if(dat[z / 4]){
				printf("\x1b[32m");
			}
			printf("0x%08x ",dat[z / 4]);
			printf("\x1b[0m");
			if(z % 16 == 12){
				printf("\n");
			}
		}
		if(z % 16){
			printf("\n");
		}
	}
	printf("\n");
	return r;
}

char *getenv(const char *name){
	static char *(*shim_getenv)(const char *);
	char *r;

	if(shim_getenv == NULL){
		const char *msg;

		fprintf(stderr,"shimming system's getenv(2)\n");
		if((shim_getenv = dlsym(RTLD_NEXT,"getenv")) == NULL){
			fprintf(stderr,"got a NULL getenv(2)\n");
			return NULL;
		}
		if( (msg = dlerror()) ){
			fprintf(stderr,"couldn't shim getenv(2): %s\n",msg);
			return NULL;
		}
	}
	printf("getenv(\x1b[1m%s\x1b[0m) = ",name);
	if( (r = shim_getenv(name)) ){
		printf("\x1b\"[1m\"%s\x1b[0m\n",r);
	}else{
		printf("\x1b[1mNULL\x1b[0m\n");
	}
	return r;
}
