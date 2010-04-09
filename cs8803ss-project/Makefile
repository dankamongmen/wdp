.DELETE_ON_ERROR:
.PHONY: all preview profile test clean
.DEFAULT_GOAL:=test

VPDF:=evince

PRO:=cs8803ssProject
PDF:=$(PRO).pdf
TEX:=$(PRO).tex
TEXOBJS:=$(wildcard texobjs/*)
BIN:=$(addprefix out/,cudadump)
PTX:=$(addsuffix .ptx,$(BIN))
PROFDATA:=$(addsuffix .prof,$(BIN))

CUDA?=/usr/local/cuda
CUDAINC:=$(CUDA)/include
CUDALIB:=$(CUDA)/lib64

CC:=$(CUDA)/bin/nvcc
CFLAGS:=-O2 --compiler-options -W,-Werror,-Wextra,-march=native,-mtune=native -I$(CUDAINC)
CFLAGS+=--compiler-bindir=/usr/bin/gcc-4.3 --ptxas-options=-v
LFLAGS:=-lcuda --linker-options -R$(CUDALIB)
PTXFLAGS:=--ptx
#CUBINFLAGS:=-deviceemu
TAGS:=.tags

all: $(PDF) $(BIN) $(PTX)

$(TAGS): src/cudadump/cudadump.cu
	@[ -d $(@D) ] || mkdir -p $(@D)
	ctags --langmap=c:.c.cu -f $@ $^

out/cudadump: src/cudadump/cudadump.cu $(TAGS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CUBINFLAGS) $(CFLAGS) -o $@ $< $(LFLAGS)

out/cudadump.ptx: src/cudadump/cudadump.cu $(TAGS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(PTXFLAGS) $(CFLAGS) -o $@ $< $(LFLAGS)


%.pdf: %.tex %.bib $(TEXOBJS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

profile: $(PROFDATA)

PROF:=CUDA_PROFILE_LOG=$(shell pwd)/$(PROFDATA) CUDA_PROFILE=1
$(PROFDATA): test
	@[ -d $(@D) ] || mkdir -p $(@D)
	env $(PROF) out/cudadump
	cat $@

preview: all
	$(VPDF) $(PDF)

test: all
	timelimit -T 1 -t 10 ./$(BIN)
	#./$(BIN)

clean:
	rm -rf out $(TAGS) $(wildcard *.dump)
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg)
	rm -f $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
