.DELETE_ON_ERROR:
.PHONY: all preview test clean docclean
.DEFAULT_GOAL:=preview

VPDF:=evince

OUT:=out
PRO:=nigeria
TEX:=$(PRO).tex
PDF:=$(PRO).pdf
TEXOBJS:=$(wildcard texobjs/*)

CFLAGS+=-Wall -Werror -fPIC $(DRIOFLAGS)

all: $(PDF)

%.pdf: docclean %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $(basename $@)
	bibtex $(basename $@)
	pdflatex $(basename $@)
	pdflatex $(basename $@)

preview: $(PDF)
	$(VPDF) $(PDF)

clean: docclean
	rm -rf $(OUT)

docclean:
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.out)
