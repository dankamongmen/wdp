.DELETE_ON_ERROR:

.PHONY: test default clean

.DEFAULT: default

BIN:=karma
ASM:=$(BIN).s

include ../Makefile.inc

MAKEFILE:=Makefile

#-fdelayed-branch

default: $(BIN) $(ASM) test

test:
	./$(BIN)

$(BIN): $(BIN).o
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $<

$(ASM): $(BIN).c $(MAKEFILE) $(BIN).h ../Makefile.inc
	$(CC) $(CFLAGS) -S -o $@ $<

$(BIN).o: $(BIN).c $(MAKEFILE) $(BIN).h ../Makefile.inc
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(BIN) $(BIN).o $(ASM)
