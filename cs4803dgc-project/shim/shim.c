#include <stdio.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>

int ioctl(int fd,int req,uintptr_t op){//,unsigned o1,unsigned o2){
	static int (*shim_ioctl)(int,int,uintptr_t,int,int);
	static int (*shim_ioctl3)(int,int,uintptr_t);
	int r;

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
	/*if(o1){
		printf("5-IOCTL[%d]: 0x%x (%x, %x, %x)\n",fd,req,op,o1,o2);
		r = shim_ioctl(fd,req,op,o1,o2);
	}else{*/
		printf("3-IOCTL[%d]: 0x%x (%jx)\n",fd,req,op);
		r = shim_ioctl3(fd,req,op);
	//}
	// r = 0;
	printf("RESULT: %d\n",r);
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
