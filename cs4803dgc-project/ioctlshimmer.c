#include <stdio.h>
#include <dlfcn.h>

int __wrap_ioctl(int fd,int req,...){
	int (*realioctl)(int,int,...);

	printf("got fd %d and req 0x%x\n",fd,req);
	if((realioctl = dlsym(RTLD_NEXT,"ioctl")) == NULL){
		fprintf(stderr,"Couldn't get the real ioctl(2)\n");
		return -1;
	}
	return realioctl(fd,req);
}
