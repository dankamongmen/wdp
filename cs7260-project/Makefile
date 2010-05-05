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
IMG:=$(addsuffix .pdf,$(addprefix $(OUT)/,$(basename $(wildcard *.pl))))

CFLAGS:=-O2 -W -Wall -Werror -Wextra -march=native -mtune=native

all: $(IMG) $(PDF) $(BIN)

%.pdf: %.tex %.bib $(IMG) $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $<
	#bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

$(OUT)/%.pdf: %.pl $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	gnuplot $<

preview: test
	$(VPDF) $(PDF)

test: all
	# FIXME no binaries yet

clean:
	rm -rf $(OUT)
	rm -f $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
