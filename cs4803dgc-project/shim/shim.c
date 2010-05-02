#include <stdio.h>
#include <errno.h>
#include <dlfcn.h>

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
