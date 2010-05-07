.DELETE_ON_ERROR:
.PHONY: all preview test clean docclean
.DEFAULT_GOAL:=preview

VPDF:=evince

OUT:=out
PRO:=cs8803dcproject
TEX:=$(PRO).tex
PDF:=$(PRO).pdf
TEXOBJS:=$(wildcard texobjs/*)
LIB:=$(OUT)/libdaytripper.so

DRIO:=$(HOME)/local/dynamorio-read-only/build
DRIOFLAGS:=-I$(DRIO)/include -DLINUX -DX86_64 -nostdlib -nostartfiles -nodefaultlibs
DRIOFLAGS:=$(DRIOFLAGS) -Xlinker -T -Xlinker $(DRIO)/api/samples/ldscript
DRIOFLAGS:=$(DRIOFLAGS)

CFLAGS+=-Wall -Werror -fPIC $(DRIOFLAGS)

all: $(LIB) $(PDF)

$(LIB): daytripper.c
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS)  -Wl,-soname,$(@F) -shared -o $@ $< $(LFLAGS)

%.pdf: docclean %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $(basename $@)
	bibtex $(basename $@)
	pdflatex $(basename $@)
	pdflatex $(basename $@)

preview: $(PDF)
	$(VPDF) $(PDF)

test: $(LIB)
	$(DRIO)/bin/drrun -v -client $(shell pwd)/$(LIB) 0 "" /bin/ls

clean: docclean
	rm -rf $(OUT)

docclean:
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
