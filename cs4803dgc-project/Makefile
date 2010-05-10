.DELETE_ON_ERROR:
.PHONY: all preview clean
.DEFAULT_GOAL:=preview

VPDF:=evince

OUT:=out
PRO:=cs4803dgcfinal
TEX:=$(PRO).tex
PDF:=$(PRO).pdf
TEXOBJS:=$(wildcard texobjs/*)

all: $(PDF)

%.pdf: %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

preview: all
	$(VPDF) $(PDF)

clean:
	rm -rf $(OUT) $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
