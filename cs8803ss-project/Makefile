.DELETE_ON_ERROR:
.PHONY: all preview test clean
.DEFAULT_GOAL:=test

VPDF:=evince

PRO:=cs8803ssProject
PDF:=$(PRO).pdf
TEX:=$(PRO).tex
TEXOBJS:=$(wildcard texobjs/*)
BIN:=$(addprefix out/,cudamemdump)
PTX:=$(addsuffix .ptx,$(BIN))

CUDA?=$(HOME)/local/cuda/
CUDAINC:=$(CUDA)/include
CUDALIB:=$(CUDA)/lib64

CC:=$(CUDA)/bin/nvcc
CFLAGS:=-O2 --compiler-options -W,-Werror,-Wextra,-march=native,-mtune=native -I$(CUDAINC)
CFLAGS+=--compiler-bindir=/usr/bin/gcc-4.3 --ptxas-options=-v
LFLAGS:=--linker-options -R$(CUDALIB) -L$(CUDALIB) -lcuda -lcudart
PTXFLAGS:=--ptx

all: $(PDF) $(PTX) $(BIN)

out/cudamemdump: src/cudamemdump.cu $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $< $(LFLAGS)

out/cudamemdump.ptx: src/cudamemdump.cu $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(PTXFLAGS) $(CFLAGS) -o $@ $< $(LFLAGS)


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
	rm -rf out
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg)
	rm -f $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
