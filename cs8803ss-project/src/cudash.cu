#include <ctype.h>
#include <stdlib.h>
#include <readline/history.h>
#include <readline/readline.h>

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
cudash_help(const char *c,const char *cmdline){
}

static struct {
	const char *cmd;
	cudashfxn fxn;
} cmdtable[] = {
	{ "help",	cudash_help,	},
	{ "exit",	cudash_quit,	},
	{ "quit",	cudash_quit,	},
	{ NULL,		NULL,		}
};

static int
run_command(const char *cmd){
	typeof(*cmdtable) *tptr;
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
		for(tptr = cmdtable ; tptr->cmd ; ++tptr){
			if(strcmp(tptr->cmd,toke) == 0){
				fxn = tptr->fxn;
				break;
			}
		}
	}
	if(fxn == NULL){
		fprintf(stderr,"Invalid command: \"%.*s\"\n",cmd - toke,toke);
		return 0;
	}
	printf("running \"%s\"(\"%s\")\n",toke,cmd);
	return fxn(toke,cmd);
}

int main(void){
	const char *prompt = "cudash> ";
	char *rln;

	// FIXME initalize CUDA on all devices
	while( (rln = readline(prompt)) ){
		// An empty string ought neither be saved to history nor run.
		if(strcmp("",rln)){
			if(add_to_history(rln)){
				fprintf(stderr,"Error adding input to history. Exiting.\n");
				free(rln);
				exit(EXIT_FAILURE);
			}
			if(run_command(rln)){
				fprintf(stderr,"Exception while running command. Exiting.\n");
				free(rln);
				exit(EXIT_FAILURE);
			}
		}
		free(rln);
	}
	exit(EXIT_SUCCESS);
}
