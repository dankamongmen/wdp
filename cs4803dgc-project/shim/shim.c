#include <stdio.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>

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
	r = shim_ioctl3(fd,req,op);
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
	printf("\x1b[1m\x1b[32m\tRESULT: %d\x1b[0m\n",r);
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
	printf("GETENV: %s\n",name);
	if( (r = shim_getenv(name)) ){
		printf("RESULT: %s\n",r);
	}
	return r;
}
