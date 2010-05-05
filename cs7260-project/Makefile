.DELETE_ON_ERROR:
.PHONY: all preview test clean
# FIXME switch to test once we have binaries
.DEFAULT_GOAL:=preview

VPDF:=evince
VSVG:=rsvg-view

OUT:=out
PRO:=cs7260final
TEX:=$(PRO).tex
PDF:=$(PRO).pdf
TEXOBJS:=$(wildcard texobjs/*)
SVG:=$(addsuffix .svg,$(addprefix $(OUT)/,$(basename $(wildcard *.pl))))

CFLAGS:=-O2 -W -Wall -Werror -Wextra -march=native -mtune=native

all: $(SVG) $(PDF) $(BIN)

%.pdf: %.tex %.bib $(SVG) $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $<
	#bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

$(OUT)/%.svg: %.pl $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	gnuplot $<

preview: test
	$(VSVG) $(SVG)
	$(VPDF) $(PDF)

test: all
	# FIXME no binaries yet

clean:
	rm -rf $(OUT)
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
