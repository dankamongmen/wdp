.DELETE_ON_ERROR:
.PHONY: all preview profile test clean
# FIXME switch to test once we have binaries
.DEFAULT_GOAL:=preview

VPDF:=evince

PRO:=cs7260final
TEX:=$(PRO).tex
PDF:=$(PRO).pdf
TEXOBJS:=$(wildcard texobjs/*)
#BIN:=$(addprefix out/,cudamemdump)

CFLAGS:=-O2 -W -Wall -Werror -Wextra -march=native -mtune=native

all: $(PDF) $(BIN)

%.pdf: %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $<
	#bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

preview: test
	$(VPDF) $(PDF)

test: all
	# FIXME no binaries yet

clean:
	rm -rf out
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg)
	rm -f $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
