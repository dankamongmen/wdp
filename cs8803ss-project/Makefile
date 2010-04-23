.DELETE_ON_ERROR:
.PHONY: all bin doc ptx preview profile test fulltest clean
.DEFAULT_GOAL:=test

include Makefile.local

VPDF:=evince

SRC:=src
OUT:=out
PRO:=cs8803ssProject
PDF:=$(PRO).pdf
TEX:=$(PRO).tex
TEXOBJS:=$(wildcard texobjs/*)
CUDADUMP:=out/cudadump
CUDAMINIMAL:=out/cudaminimal
CUDAPINNER:=out/cudapinner
CUDAQUIRKY:=out/cudaquirky
CUDARANGER:=out/cudaranger
CUDASPAWNER:=out/cudaspawner
CUDASTUFFER:=out/cudastuffer
CSRC:=$(wildcard src/*.c)
CUDASRC:=$(wildcard src/*.cu)
BIN:=$(addprefix out/,$(basename $(notdir $(CSRC) $(CUDASRC))))
PTX:=$(addsuffix .ptx,$(addprefix out/,$(basename $(notdir $(CUDASRC)))))
PROFDATA:=$(addsuffix .prof,$(BIN))

CUDA?=/usr/
CUDAINC?=$(CUDA)/include/cuda
CUDART?=$(HOME)/local/cuda/
CUDARTLIB:=$(CUDART)/lib64

NVCC?=$(CUDA)/bin/nvcc
CFLAGS:=-g -ggdb -Wall -W -Werror -march=native -mtune=native
NCFLAGS:=-g -G --compiler-options -W,-Werror,-Wextra,-march=native,-mtune=native -I$(SRC) -I$(CUDAINC)
NCFLAGS+=--compiler-bindir=/usr/bin/gcc-4.3 --ptxas-options=-v
LFLAGS:=-lcuda
NLFLAGS:=$(LFLAGS) --linker-options -R$(CUDARTLIB)
PTXFLAGS:=--ptx
TAGS:=.tags

all: $(TAGS) doc bin ptx
       
doc: $(PDF)
	
bin: $(TAGS) $(BIN)
	
ptx: $(PTX)

$(TAGS): $(CSRC) $(CUDASRC) util/cuda8803ss.c $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	ctags --langmap=c:.c.cu.h -f $@ $^

$(OUT)/cudadump: $(OUT)/cudadump.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS)

$(OUT)/cudaminimal: $(OUT)/cudaminimal.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/cudapinner: $(OUT)/cudapinner.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/cudaquirky: $(OUT)/cudaquirky.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS)

$(OUT)/cudaranger: $(OUT)/cudaranger.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -o $@ $^ $(NLFLAGS)

$(OUT)/cudaspawner: $(OUT)/cudaspawner.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/cudastuffer: $(OUT)/cudastuffer.o $(OUT)/cuda8803ss.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

$(OUT)/%.ptx: $(SRC)/%.cu
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(PTXFLAGS) $(NCFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/%.o: $(SRC)/%.cu $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(NVCC) $(NCFLAGS) -c -o $@ $< $(LFLAGS)

$(OUT)/%.o: $(SRC)/%.c $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) -I$(SRC) -I$(CUDAINC) $(CFLAGS) -c -o $@ $< $(LFLAGS)

$(OUT)/%.o: util/%.c $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) -I$(SRC) -I$(CUDAINC) $(CFLAGS) -c -o $@ $< $(LFLAGS)

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

preview: doc
	$(VPDF) $(PDF)

CUDADEVNO?=0
test: bin
	./$(CUDARANGER) $(CUDADEVNO) 0x110000 0x111000
	./$(CUDASTUFFER) 0
	./$(CUDASPAWNER) $(CUDADEVNO) 0x100000
	./$(CUDAQUIRKY)
	! ./$(CUDARANGER) $(CUDADEVNO) 0 0x100000000

fulltest: test
	./$(CUDADUMP)
	./$(CUDAPINNER) 0
	! ./$(CUDARANGER) $(CUDADEVNO) 0 0x100000000

clean:
	rm -rf out $(TAGS) $(wildcard *.dump) $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
