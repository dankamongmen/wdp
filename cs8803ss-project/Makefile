.DELETE_ON_ERROR:
.PHONY: all test clean
.DEFAULT_GOAL:=test

VPDF:=evince

CFLAGS:=-Wall -W -Werror -O2

PRO:=cs8803ssProject
PDF:=$(PRO).pdf
TEX:=$(PRO).tex
TEXOBJS:=$(wildcard texobjs/*)
BIN:=cudamemdump

CUDA?=$(HOME)/local/cuda/
CUDAINC:=$(CUDA)/include
CUDALIB:=$(CUDA)/lib64

CFLAGS:=-Wall -W -O2 -march=native -mtune=native -Werror -Wextra -I$(CUDAINC)
LFLAGS:=-Wl,-R$(CUDALIB) -L$(CUDALIB) -lcuda -lcudart

all: $(BIN) $(PDF)

cudamemdump: src/cudamemdump.c $(MAKEFILES)
	$(CC) $(CFLAGS) -o $@ $< $(LFLAGS)

%.pdf: %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

test: all
	./$(BIN)
	$(VPDF) $(PDF)

clean:
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.out)
	rm -f $(BIN) $(PDF)
