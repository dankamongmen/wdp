#include <errno.h>
#include <ctype.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <readline/history.h>
#include <readline/readline.h>
#include "cuda8803ss.h"

typedef struct cudamap {
	uintptr_t base;
	size_t s;		// only what we asked for, not actually got
	struct cudamap *next;
	void *maps;		// only for on-device mappings of host memory.
				// otherwise, equal to MAP_FAILED.
} cudamap;

typedef struct cudadev {
	char *devname;
	int devno;
	struct cudadev *next;
	int major,minor,warpsz,mpcount;
	CUcontext ctx;
	cudamap *map;
} cudadev;

static cudamap *maps;		// FIXME ought be per-card; we're overloading
static unsigned cudash_child;
static cudadev *devices,*curdev;

static int
add_to_history(const char *rl){
	if(strcmp(rl,"") == 0){
		return 0;
	}
	add_history(rl); // FIXME error check?
	return 0;
}

static cudamap *
create_cuda_map(uintptr_t p,size_t s,void *targ){
	cudamap *r;

	if((r = (cudamap *)malloc(sizeof(*r))) == NULL){
		fprintf(stderr,"Couldn't allocate map (%s)\n",strerror(errno));
		return NULL;
	}
	r->base = p;
	r->s = s;
	r->maps = targ;
	return r;
}

typedef int (*cudashfxn)(const char *,const char *);

static int
cudash_quit(const char *c,const char *cmdline){
	if(strcmp(cmdline,"")){
		fprintf(stderr,"Command line following %s; did you really mean to quit?\n",c);
		return 0;
	}
	if(!cudash_child){
		printf("Thank you for using the CUDA shell. Have a very CUDA day.\n");
	}
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
create_ctx_mapofmap(cudamap **m,uintptr_t p,size_t size,void *targ){
	cudamap *cm;

	if((cm = create_cuda_map(p,size,targ)) == NULL){
		return 0;
	}
	while(*m){
		if(cm->base <= (*m)->base){
			break;
		}
		m = &(*m)->next;
	}
	cm->next = *m;
	*m = cm;
	return 0;
}

static inline int
create_ctx_map(cudamap **m,uintptr_t p,size_t size){
	return create_ctx_mapofmap(m,p,size,MAP_FAILED);
}

static int
cudash_read(const char *c,const char *cmdline){
	unsigned long long base,size;
	dim3 db(BLOCK_SIZE,1,1);
	dim3 dg(GRID_SIZE,1,1);
	CUdeviceptr res;
	CUresult cerr;
	char *ep;

	if(((base = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid base: %s\n",cmdline);
		return 0;
	}
	cmdline = ep;
	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %s\n",cmdline);
		return 0;
	}
	if(printf("Reading [0x%llx:0x%llx) (0x%llx)\n",base,base + size,size) < 0){
		return -1;
	}
	if((cerr = cuMemAlloc(&res,sizeof(uint32_t) * BLOCK_SIZE)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't allocate result array (%s)\n",strerror(errno));
		return 0;
	}
	readkernel<<<dg,db>>>((unsigned *)base,(unsigned *)(base + size),
				(uint32_t *)res);
	// FIXME inspect result array
	if((cerr = cuMemFree(res)) != CUDA_SUCCESS
			|| (cerr = cuCtxSynchronize()) != CUDA_SUCCESS){
		if(fprintf(stderr,"Error reading memory (%d)\n",cerr) < 0){
			return -1;
		}
	}else{
		if(printf("Successfully read memory.\n") < 0){
			return -1;
		}
	}
	return 0;
}

static int
cudash_write(const char *c,const char *cmdline){
	unsigned long long base,size;
	CUresult cerr;
	char *ep;

	if(((base = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid base: %s\n",cmdline);
		return 0;
	}
	cmdline = ep;
	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %s\n",cmdline);
		return 0;
	}
	if(printf("Writing [0x%llx:0x%llx) (0x%llx)\n",base,base + size,size) < 0){
		return -1;
	}
	// FIXME write it
	if((cerr = cuCtxSynchronize()) != CUDA_SUCCESS){
		fprintf(stderr,"Error writing memory (%d)\n",cerr);
	}else{
		if(printf("Successfully wrote memory.\n") < 0){
			return -1;
		}
	}
	return 0;
}

static int
cudash_alloc(const char *c,const char *cmdline){
	unsigned long long size;
	CUdeviceptr p;
	CUresult cerr;
	char *ep;

	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %s\n",cmdline);
		return 0;
	}
	if((cerr = cuMemAlloc(&p,size)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't allocate %llub (%d)\n",size,cerr);
		return 0;
	}
	if(create_ctx_map(&curdev->map,p,size)){
		cuMemFree(p);
		return 0;
	}
	printf("Allocated %llub @ %p\n",size,p);
	return 0;
}

static int
cudash_pin(const char *c,const char *cmdline){
	unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
	unsigned long long size;
	CUdeviceptr cd;
	CUresult cerr;
	char *ep;
	void *p;

	if(((size = strtoull(cmdline,&ep,0)) == ULONG_MAX && errno == ERANGE)
			|| cmdline == ep){
		fprintf(stderr,"Invalid size: %s\n",cmdline);
		return 0;
	}
	if((cerr = cuMemHostAlloc(&p,size,flags)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't host-allocate %llub (%d)\n",size,cerr);
		return 0;
	}
	if(create_ctx_map(&maps,(uintptr_t)p,size)){
		cuMemFreeHost(p);
		return 0;
	}
	printf("Allocated %llub host memory @ %p\n",size,p); // FIXME adjust map
	// FIXME map into each card's memory space, not just current's
	if((cerr = cuMemHostGetDevicePointer(&cd,p,curdev->devno)) != CUDA_SUCCESS){
		fprintf(stderr,"Couldn't map %llub @ %p on dev %d (%d)\n",
				size,p,curdev->devno,cerr);
		cuMemFreeHost(p);
		// FIXME need to extract from host map list, previous devices
		return 0;
	}
	if(create_ctx_mapofmap(&curdev->map,cd,size,p)){
		cuMemFreeHost(p);
		// FIXME need to extract from host map list, previous devices
		return 0;
	}
	printf("Mapped %llub into card %d @ %p\n",size,0,cd);
	return 0;
}

static int
cudash_fork(const char *c,const char *cmdline){
	pid_t pid;

	if(fflush(stdout) || fflush(stderr)){
		fprintf(stderr,"Couldn't flush output (%s?)\n",strerror(errno));
		return -1;
	}
	if((pid = fork()) < 0){
		fprintf(stderr,"Couldn't fork (%s?)\n",strerror(errno));
		return -1;
	}else if(pid == 0){
		cudash_child = 1;
		printf("Type \"exit\" to leave this child (PID %ju)\n",(uintmax_t)getpid());
		return 0;
	}else{
		int status;

		waitpid(pid,&status,0); // FIXME check result code
		printf("Returning to parent shell (PID %ju)\n",(uintmax_t)getpid());
	}
	return 0;
}

static int
cudash_exec(const char *c,const char *cmdline){
	pid_t pid;

	if(fflush(stdout) || fflush(stderr)){
		fprintf(stderr,"Couldn't flush output (%s?)\n",strerror(errno));
		return -1;
	}
	if((pid = fork()) < 0){
		fprintf(stderr,"Couldn't fork (%s?)\n",strerror(errno));
		return -1;
	}else if(pid == 0){
		while(isspace(*cmdline)){
			++cmdline;
		}{
			// FIXME tokenize that bitch
			char * const argv[] = { strdup(cmdline), NULL };
			if(execvp(cmdline,argv)){
				fprintf(stderr,"Couldn't launch %s (%s)\n",cmdline,strerror(errno));
			}
		}
		exit(EXIT_FAILURE);
	}else{
		int status;

		waitpid(pid,&status,0); // FIXME check result code
	}
	return 0;
}

static int
cudash_maps(const char *c,const char *cmdline){
	cudadev *d;
	cudamap *m;

	for(m = maps ; m ; m = m->next){
		if(printf("(host) %10zu (0x%08x) @ 0x%012jx\n",
				m->s,m->s,(uintmax_t)m->base) < 0){
			return -1;
		}
	}
	for(d = devices ; d ; d = d->next){
		for(m = d->map ; m ; m = m->next){
			if(printf("(%4d) %10zu (0x%08x) @ 0x%012jx",
					d->devno,m->s,m->s,(uintmax_t)m->base) < 0){
				return -1;
			}
			if(m->maps != MAP_FAILED){
				if(printf(" (maps %012p)",m->maps) < 0){
					return -1;
				}
			}
			if(printf("\n") < 0){
				return -1;
			}
		}
	}
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
	{ "exec",	cudash_exec,	"fork, and exec a binary",	},
	{ "exit",	cudash_quit,	"exit the CUDA shell",	},
	{ "fork",	cudash_fork,	"fork a child cudash",	},
	{ "help",	cudash_help,	"help on the CUDA shell and commands",	},
	{ "maps",	cudash_maps,	"display CUDA memory tables",	},
	{ "pin",	cudash_pin,	"pin and map host memory",	},
	{ "quit",	cudash_quit,	"exit the CUDA shell",	},
	{ "read",	cudash_read,	"read device memory in CUDA",	},
	{ "write",	cudash_write,	"write device memory in CUDA",	},
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
free_maps(cudamap *m){
	while(m){
		cudamap *tm = m;

		m = tm->next;
		free(tm);
	}
}

static void
free_devices(cudadev *d){
	while(d){
		cudadev *t = d;

		d = d->next;
		free(t->devname);
		free_maps(t->map);
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
		c->map = NULL;
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
	curdev = devices;
	while( (rln = readline(prompt)) ){
		// An empty string ought neither be saved to history nor run.
		if(strcmp("",rln)){
			if(add_to_history(rln)){
				fprintf(stderr,"Error adding input to history. Exiting.\n");
				free(rln);
				break;
			}
			if(run_command(rln)){
				free(rln);
				break;
			}
		}
		free(rln);
	}
	free_devices(devices);
	free_maps(maps);
	exit(EXIT_SUCCESS);
}
