#include <errno.h>
#include <ctype.h>
#include <stdlib.h>
#include <readline/history.h>
#include <readline/readline.h>
#include "cuda8803ss.h"

typedef struct cudadev {
	char *devname;
	int devno;
	struct cudadev *next;
	int major,minor,warpsz,mpcount;
	CUcontext ctx;
} cudadev;

static cudadev *devices;

static int
add_to_history(const char *rl){
	if(strcmp(rl,"") == 0){
		return 0;
	}
	add_history(rl); // FIXME error check?
	return 0;
}

typedef int (*cudashfxn)(const char *,const char *);

static int
cudash_quit(const char *c,const char *cmdline){
	if(strcmp(cmdline,"")){
		fprintf(stderr,"Command line following %s; did you really mean to quit?\n",c);
		return 0;
	}
	printf("Thank you for using the CUDA shell. Have a very CUDA day.\n");
	exit(EXIT_SUCCESS);
}

static int
list_cards(void){
	cudadev *c;

	for(c = devices ; c ; c = c->next){
		if(printf("Card %d: %s, capability %d.%d, %d MPs\n",
			c->devno,c->devname,c->major,c->minor,c->mpcount) < 0){
			return -1;
		}
		// FIXME more detail
	}
	return 0;
}

static int
cudash_cards(const char *c,const char *cmdline){
	if(strcmp(cmdline,"")){
		fprintf(stderr,"%s doesn't support options\n");
		return 0;
	}
	return list_cards();
}

static int
cudash_alloc(const char *c,const char *cmdline){
	unsigned long long size;
	CUdeviceptr p;
	CUresult cerr;
	char *ep;

	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %.*s\n",ep - cmdline,cmdline);
		return 0;
	}
	if((cerr = cuMemAlloc(&p,size)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't allocate %llub (%d)\n",size,cerr);
		return 0;
	}
	printf("Allocated %llub @ %p\n",size,p); // FIXME adjust map
	return 0;
}

static int cudash_help(const char *,const char *);

static const struct {
	const char *cmd;
	cudashfxn fxn;
	const char *help;
} cmdtable[] = {
	{ "alloc",	cudash_alloc,	"allocate device memory",	},
	{ "cards",	cudash_cards,	"list devices supporting CUDA",	},
	{ "exit",	cudash_quit,	"exit the CUDA shell",	},
	{ "help",	cudash_help,	"help on the CUDA shell and commands",	},
	{ "quit",	cudash_quit,	"exit the CUDA shell",	},
	{ NULL,		NULL,		NULL,	}
};

static typeof(*cmdtable) *
lookup_command(const char *c,size_t n){
	typeof(*cmdtable) *tptr;

	for(tptr = cmdtable ; tptr->cmd ; ++tptr){
		if(strncmp(tptr->cmd,c,n) == 0 && strlen(tptr->cmd) == n){
			return tptr;
		}
	}
	return NULL;
}

static int
list_commands(void){
	typeof(*cmdtable) *t;

	for(t = cmdtable ; t->cmd ; ++t){
		if(printf("%s: %s\n",t->cmd,t->help) < 0){
			return -1;
		}
	}
	return 0;
}

static int
cudash_help(const char *c,const char *cmdline){
	if(strcmp(cmdline,"") == 0){
		return list_commands();
	}else{
		typeof(*cmdtable) *tptr;

		while(isspace(*cmdline)){
			++cmdline;
		}
		// FIXME extract first token
		if((tptr = lookup_command(cmdline,strlen(cmdline))) == NULL){
			if(printf("No help is available for \"%s\".\n",cmdline) < 0){
				return -1;
			}
		}else{
			if(printf("%s: %s\n",tptr->cmd,tptr->help) < 0){
				return -1;
			}
		}
	}
	return 0;
}

static int
run_command(const char *cmd){
	cudashfxn fxn = NULL;
	const char *toke;

	while(isspace(*cmd)){
		++cmd;
	}
	toke = cmd;
	while(isalnum(*cmd)){
		++cmd;
	}
	if(cmd != toke){
		typeof(*cmdtable) *tptr;

		if( (tptr = lookup_command(toke,cmd - toke)) ){
			fxn = tptr->fxn;
		}
	}
	if(fxn == NULL){
		fprintf(stderr,"Invalid command: \"%.*s\"\n",cmd - toke,toke);
		return 0;
	}
	return fxn(toke,cmd);
}

static void
free_devices(cudadev *d){
	while(d){
		cudadev *t = d;

		d = d->next;
		free(t->devname);
		cuCtxDestroy(t->ctx);
		free(t);
	}
}

static int
id_cudadev(cudadev *c){
	struct cudaDeviceProp dprop;
	CUdevice d;
	int cerr;

	if((cerr = cuDeviceGet(&d,c->devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't query device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	if((cerr = cudaGetDeviceProperties(&dprop,d)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't query device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	cerr = cuDeviceGetAttribute(&c->warpsz,CU_DEVICE_ATTRIBUTE_WARP_SIZE,d);
	if(cerr != CUDA_SUCCESS || c->warpsz <= 0){
		fprintf(stderr,"Couldn't get warp size for device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	cerr = cuDeviceGetAttribute(&c->mpcount,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,d);
	if(cerr != CUDA_SUCCESS || c->mpcount <= 0){
		fprintf(stderr,"Couldn't get MP count for device %d (%d)\n",c->devno,cerr);
		return -1;
	}
	if((cerr = cuDeviceComputeCapability(&c->major,&c->minor,d)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get compute capability for device %d (%d)\n",c->devno,cerr);
		return -1;
	}
#define CUDASTRLEN 80
	if((c->devname = (char *)malloc(CUDASTRLEN)) == NULL){
		fprintf(stderr,"Couldn't allocate %zub (%s?)\n",CUDASTRLEN,strerror(errno));
		return -1;
	}
	if((cerr = cuDeviceGetName(c->devname,CUDASTRLEN,d)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get name for device %d (%d)\n",c->devno,cerr);
		free(c->devname);
		return -1;
	}
#undef CUDASTRLEN
	if((cerr = cuCtxCreate(&c->ctx,CU_CTX_MAP_HOST,d)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't get context for device %d (%d)\n",c->devno,cerr);
		free(c->devname);
		return -1;
	}
	return 0;
}

static int
make_devices(int count){
	cudadev *chain = NULL;

	while(count--){
		cudadev *c;

		if((c = (cudadev *)malloc(sizeof(*c))) == NULL){
			free_devices(chain);
			return -1;
		}
		c->devno = count;
		if(id_cudadev(c)){
			free_devices(chain);
			free(c);
			return -1;
		}
		c->next = chain;
		chain = c;
	}
	devices = chain;
	return 0;
}

int main(void){
	const char *prompt = "cudash> ";
	char *rln = NULL;
	int count;

	if(init_cuda_alldevs(&count)){
		exit(EXIT_FAILURE);
	}
	if(make_devices(count)){
		exit(EXIT_FAILURE);
	}
	while( (rln = readline(prompt)) ){
		// An empty string ought neither be saved to history nor run.
		if(strcmp("",rln)){
			if(add_to_history(rln)){
				fprintf(stderr,"Error adding input to history. Exiting.\n");
				goto err;
			}
			if(run_command(rln)){
				fprintf(stderr,"Exception while running command. Exiting.\n");
				goto err;
			}
		}
		free(rln);
	}
	free_devices(devices);
	exit(EXIT_SUCCESS);

err:
	free(rln);
	free_devices(devices);
	exit(EXIT_FAILURE);
}
