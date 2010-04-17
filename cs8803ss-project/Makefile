.DELETE_ON_ERROR:
.PHONY: all bin preview profile test clean
.DEFAULT_GOAL:=test

VPDF:=evince

SRC:=src
OUT:=out
PRO:=cs8803ssProject
PDF:=$(PRO).pdf
TEX:=$(PRO).tex
TEXOBJS:=$(wildcard texobjs/*)
CUDADUMP:=out/cudadump
CUDAMINIMAL:=out/cudaminimal
CUDARANGER:=out/cudaranger
CUDASPAWNER:=out/cudaspawner
CUDASTUFFER:=out/cudastuffer
BIN:=$(CUDADUMP) $(CUDAMINIMAL) $(CUDARANGER) $(CUDASPAWNER) $(CUDASTUFFER)
PTX:=$(addsuffix .ptx,$(BIN))
PROFDATA:=$(addsuffix .prof,$(BIN))

CUDA?=/usr/local/
CUDAINC:=$(CUDA)/include/cuda

NVCC?=$(CUDA)/bin/nvcc
CFLAGS:=-O2 -Wall -W -Werror -march=native -mtune=native
NCFLAGS:=-O2 --compiler-options -W,-Werror,-Wextra,-march=native,-mtune=native -I$(SRC) -I$(CUDAINC)
NCFLAGS+=--compiler-bindir=/usr/bin/gcc-4.3 --ptxas-options=-v
LFLAGS:=-lcuda
PTXFLAGS:=--ptx
#CUBINFLAGS:=-deviceemu
TAGS:=.tags

all: $(TAGS) $(PDF) $(BIN) $(PTX)

bin: $(TAGS) $(BIN)

$(TAGS): $(SRC)/cudadump.cu $(SRC)/cudaminimal.cu $(SRC)/cudaranger.cu $(SRC)/cudaspawner.c $(SRC)/cudastuffer.cu $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	ctags --langmap=c:.c.cu.h -f $@ $^

$(OUT)/cudadump: $(OUT)/cudadump.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(CUBINFLAGS) $(NCFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/cudaminimal: $(OUT)/cudaminimal.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(CUBINFLAGS) $(NCFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/cudaranger: $(OUT)/cudaranger.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(CUBINFLAGS) $(NCFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/cudaspawner: $(OUT)/cudaspawner.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/cudastuffer: $(OUT)/cudastuffer.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(CUBINFLAGS) $(NCFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/%.ptx: $(SRC)/%.cu
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(PTXFLAGS) $(NCFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/%.o: $(SRC)/%.cu $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(CUBINFLAGS) $(NCFLAGS) -c -o $@ $< $(LFLAGS)

$(OUT)/%.o: $(SRC)/%.c
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) -I$(CUDAINC) $(CFLAGS) -c -o $@ $< $(LFLAGS)

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
	./$(CUDARANGER) 0 0x110000 0x111000
	./$(CUDASTUFFER) 0
	./$(CUDASPAWNER) 0 0x100000
	./$(CUDADUMP)
	! ./$(CUDARANGER) 0 0 0x100000000

clean:
	rm -rf out $(TAGS) $(wildcard *.dump)
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg)
	rm -f $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
