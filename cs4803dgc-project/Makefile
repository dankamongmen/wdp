.DELETE_ON_ERROR:
.PHONY: all preview test clean
.DEFAULT_GOAL:=test

VPDF:=evince

OUT:=out
PRO:=cs4803dgcfinal
TEX:=$(PRO).tex
PDF:=$(PRO).pdf
LIB:=$(OUT)/libcudest.so
INC:=$(wildcard src/*.h)
TEXOBJS:=$(wildcard texobjs/*)
SRC:=$(notdir $(wildcard src/*.c))
OBJ:=$(addsuffix .o,$(addprefix $(OUT)/,$(basename $(SRC))))
TAGS:=.tags
TAGBIN:=ctags

IFLAGS:=-Isrc
CFLAGS:=-pipe $(IFLAGS) -fPIC -O2 -Wall -W -Werror -march=native -mtune=native

all: $(LIB) $(PDF)

$(TAGS): $(addprefix src/,$(SRC)) $(INC)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(TAGBIN) -f $@ $^

$(OUT)/libcudest.so: $(OBJ)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LFLAGS)
	
$(OUT)/%.o: src/%.c $(INC) $(TAGS)
	@[ -d $(@D) ] || mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

%.pdf: %.tex %.bib $(TEXOBJS) $(MAKEFILES)
	@[ -d $(@D) ] || mkdir -p $(@D)
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

preview: test
	$(VPDF) $(PDF)

test: all
	# FIXME no binaries yet

clean:
	rm -rf $(OUT) $(wildcard *.aux) $(wildcard *.bbl) $(wildcard *.blg) $(wildcard *.log) $(wildcard *.pdf) $(wildcard *.out)
