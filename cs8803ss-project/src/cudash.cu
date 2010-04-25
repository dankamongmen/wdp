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
list_cards(void){
	// FIXME
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

static int cudash_help(const char *,const char *);

static const struct {
	const char *cmd;
	cudashfxn fxn;
	const char *help;
} cmdtable[] = {
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
		if(printf(" %s: %s\n",t->cmd,t->help) < 0){
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
			if(printf(" %s: %s\n",tptr->cmd,tptr->help) < 0){
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
