.DELETE_ON_ERROR:
.PHONY: all preview test clean docclean
# FIXME switch to test once we have binaries
.DEFAULT_GOAL:=preview

VPDF:=evince

PRO:=cs8803dcproject
TEX:=$(PRO).tex
PDF:=$(PRO).pdf
TEXOBJS:=$(wildcard texobjs/*)
LIB:=libdaytripper.so

DRIO:=$(HOME)/local/dynamorio-read-only/build
DRIOFLAGS:=-I$(DRIO)/include -DLINUX -DX86_64 -nostdlib -nostartfiles -nodefaultlibs
DRIOFLAGS:=$(DRIOFLAGS) -Xlinker -T -Xlinker $(DRIO)/api/samples/ldscript
DRIOFLAGS:=$(DRIOFLAGS) -Wl,-soname,$(LIB)

CFLAGS+=-Wall -Werror -fPIC $(DRIOFLAGS)

all: $(LIB) $(PDF)

$(LIB): daytripper.c
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -o $@ $< $(LFLAGS)

%.pdf: docclean %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $(basename $@)
	bibtex $(basename $@)
	pdflatex $(basename $@)
	pdflatex $(basename $@)

preview: test
	$(VPDF) $(PDF)

test: all
	$(DRIO)/bin/drrun -client $(shell pwd)/$(LIB) 0 "" /bin/ls

clean: docclean
	rm -f $(LIB)

docclean:
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
