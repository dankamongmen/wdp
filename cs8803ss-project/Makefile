.DELETE_ON_ERROR:
.PHONY: all preview profile test clean
.DEFAULT_GOAL:=test

VPDF:=evince

SRC:=src
OUT:=out
PRO:=cs8803ssProject
PDF:=$(PRO).pdf
TEX:=$(PRO).tex
TEXOBJS:=$(wildcard texobjs/*)
CUDADUMP:=out/cudadump
CUDARANGER:=out/cudaranger
CUDASANITY:=out/cudasanity
CUDASITTER:=out/cudasitter
BIN:=$(CUDADUMP) $(CUDARANGER) $(CUDASANITY)
PTX:=$(addsuffix .ptx,$(BIN))
PROFDATA:=$(addsuffix .prof,$(BIN))

CUDA?=/usr/local/cuda
CUDAINC:=$(CUDA)/include
CUDALIB:=$(CUDA)/lib64

CC:=$(CUDA)/bin/nvcc
CFLAGS:=-O2 --compiler-options -W,-Werror,-Wextra,-march=native,-mtune=native -I$(SRC) -I$(CUDAINC)
CFLAGS+=--compiler-bindir=/usr/bin/gcc-4.3 --ptxas-options=-v
LFLAGS:=-lcuda --linker-options -R$(CUDALIB)
PTXFLAGS:=--ptx
#CUBINFLAGS:=-deviceemu
TAGS:=.tags

all: $(TAGS) $(PDF) $(BIN) $(PTX)

$(TAGS): $(SRC)/cudadump.cu $(SRC)/cudaranger.cu $(SRC)/cudasanity.cu $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	ctags --langmap=c:.c.cu.h -f $@ $^

$(OUT)/cudadump: $(OUT)/cudadump.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CUBINFLAGS) $(CFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/cudaranger: $(OUT)/cudaranger.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CUBINFLAGS) $(CFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/cudasanity: $(OUT)/cudasanity.o
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CUBINFLAGS) $(CFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/%.ptx: $(SRC)/%.cu
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(PTXFLAGS) $(CFLAGS) -o $@ $< $(LFLAGS)

$(OUT)/%.o: $(SRC)/%.cu $(SRC)/cuda8803ss.h
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CUBINFLAGS) $(CFLAGS) -c -o $@ $< $(LFLAGS)

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
	#timelimit -T 1 -t 10 ./$(CUDADUMP)
	./$(CUDARANGER) 0 0x110000 0x111000
	./$(CUDADUMP)
	! ./$(CUDARANGER) 0 0 0x100000000

clean:
	rm -rf out $(TAGS) $(wildcard *.dump)
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg)
	rm -f $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
