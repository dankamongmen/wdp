.DELETE_ON_ERROR:
.PHONY: all preview test clean
# FIXME switch to test once we have binaries
.DEFAULT_GOAL:=preview

VPDF:=evince

PRO:=cs8803dcproject
TEX:=$(PRO).tex
PDF:=$(PRO).pdf
TEXOBJS:=$(wildcard texobjs/*)

all: $(PDF)

%.pdf: clean %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $(basename $@)
	bibtex $(basename $@)
	pdflatex $(basename $@)
	pdflatex $(basename $@)

preview: test
	$(VPDF) $(PDF)

test: all
	# FIXME no binaries yet

clean:
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
