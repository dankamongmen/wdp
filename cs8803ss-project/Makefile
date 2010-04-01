.DELETE_ON_ERROR:
.PHONY: all preview test clean
.DEFAULT_GOAL:=test

VPDF:=evince

PRO:=cs8803ssProject
PDF:=$(PRO).pdf
TEX:=$(PRO).tex
TEXOBJS:=$(wildcard texobjs/*)
BIN:=cudamemdump

CUDA?=$(HOME)/local/cuda/
CUDAINC:=$(CUDA)/include
CUDALIB:=$(CUDA)/lib64

CC:=$(CUDA)/bin/nvcc
CFLAGS:=-O2 --compiler-options -W,-Werror,-march=native,-mtune=native -I$(CUDAINC)
CFLAGS+=--compiler-bindir=/usr/bin/gcc-4.3
LFLAGS:=--linker-options -R$(CUDALIB) -L$(CUDALIB) -lcuda -lcudart

all: $(PDF) $(BIN)

cudamemdump: src/cudamemdump.cu $(MAKEFILES)
	$(CC) $(CFLAGS) -o $@ $< $(LFLAGS)

%.pdf: %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

preview: test
	$(VPDF) $(PDF)

test: all
	./$(BIN)

clean:
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.out)
	rm -f $(BIN) $(PDF)
