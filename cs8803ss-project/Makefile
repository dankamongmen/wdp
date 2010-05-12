.DELETE_ON_ERROR:
.PHONY: all doc preview clean
.DEFAULT_GOAL:=preview

VPDF:=evince

SRC:=src
PRO:=cs8803ssProject
PDF:=$(PRO).pdf
TEX:=$(PRO).tex
TEXOBJS:=$(wildcard texobjs/*)

all: doc
       
doc: $(PDF)
	
%.pdf: %.tex %.bib $(TEXOBJS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

preview: doc
	$(VPDF) $(PDF)

clean:
	rm -rf $(wildcard *.dump) $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
