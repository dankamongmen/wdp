.DELETE_ON_ERROR:
.PHONY: all test clean
.DEFAULT_GOAL:=test

VPDF:=evince

CFLAGS:=-Wall -W -Werror -O2

PRO:=cs8803ssProject
PROP:=cs8803ssProposal
PDF:=$(PRO).pdf $(PROP).pdf
TEX:=$(PRO).tex $(PROP).tex
TEXOBJS:=$(wildcard texobjs/*)

all: $(PDF)

$(PDF): $(TEX) $(TEXOBJS)
	pdflatex $<
	pdflatex $<

test: all
	$(VPDF) $(PDF)

clean:
	rm -f $(wildcard *.aux) $(wildcard *.log) $(wildcard *.out) $(wildcard *.pdf)
