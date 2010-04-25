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

int main(void){
	const char *prompt = "cudash> ";
	char *rln;

	while( (rln = readline(prompt)) ){
		printf("%s\n",rln);
		if(add_to_history(rln)){
			free(rln);
			exit(EXIT_FAILURE);
		}
		free(rln);
	}
	exit(EXIT_SUCCESS);
}
