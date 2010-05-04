# Makefile for the optimize matrix multiply assignment
# 
.DELETE_ON_ERROR:

# Set the compiler flags, etc., in Makefile.inc
include Makefile.inc

# Add -DDEBUG_RUN to CPPFLAGS to cut down on the cases.
CPPFLAGS = "-DCOMPILER=\"$(CC)\"" "-DFLAGS=\"$(CFLAGS)\""

# Compile a C version (using basic_dgemm.c, in this case):
OBJS = matmul.o timing.o

OURS:=doyen
#garuda
BIN:=$(OURS) matmul matmul-blocked$(NB) matmul-rec$(NB) matmul-blas
#matmul-fortran
ASM:=asm/karma.h

.PHONY:	test default all asm display timing clean realclean

default: quicktest

MAKEFILE:=Makefile Makefile.inc

test: asm all
	@for i in $(BIN) ; do ./$$i || exit 1 ; done

quicktest: $(OURS)
	./$(OURS)

all: $(BIN)

asm: $(ASM) asm/karma.c
	@cd asm && $(MAKE)

$(OURS): $(OBJS) doyen.o
	$(CC) -o $@ $(CFLAGS) $(CPPFLAGS) $^ $(LIBS)

garuda: $(OBJS) garuda.o
	$(CC) -o $@ $(CFLAGS) $(CPPFLAGS) $^ $(LIBS)

matmul: $(OBJS) basic_dgemm.o
	$(CC) -o $@ $(CFLAGS) $(CPPFLAGS) $^ $(LIBS)

matmul-blocked$(NB): $(OBJS) blocked_dgemm$(NB).o
	$(CC) -o $@ $(CFLAGS) $(CPPFLAGS) $^ $(LIBS)

blocked_dgemm$(NB).o: blocked_dgemm.c
	$(CC) -o $@ $(CFLAGS) $(CPPFLAGS) -DBLOCK_SIZE=$(NB) -c $^

matmul-rec$(NB): $(OBJS) rec_dgemm$(NB).o
	$(CC) -o $@ $(CFLAGS) $(CPPFLAGS) $^ $(LIBS)

rec_dgemm$(NB).o: rec_dgemm.c
	$(CC) -o $@ $(CFLAGS) $(CPPFLAGS) -DBLOCK_SIZE=$(NB) -c $^

# An example of using Fortran.
matmul-fortran:	$(OBJS) fortran_dgemm.o fortran_dgemm_wrapper.o
	$(F77) -o $@ $^ $(LIBS)

matmul-blas: $(OBJS) blas_dgemm.o
	$(CC) -o $@ $^ $(LIBS) $(BLASLIB)

%.o: %.c $(OURS).h $(MAKEFILE)
	$(CC) -o $@ -c $(CFLAGS) $(CPPFLAGS) $<

doyen.s: doyen.c $(OURS).h $(ASM) $(MAKEFILE)
	$(CC) -o $@ -S $(CFLAGS) $(CPPFLAGS) $<

# ---
# This is just a suggestion on how to generate timing plots...  Feel
# free to improve on these, so long as you show MFlop/s v. matrix size.
#
# $(MM) is an optional suffix to indicate which 'matmul*' binary to
# run. The output files will have this suffix as well.
#

timing: timing$(MM).raw
	cat $< | grep 'mflop/s' | awk '{print $$2, $$4;}' > $@$(MM)

timing$(MM).raw: ./matmul$(MM)
	./matmul$(MM) 2>&1 | tee $@

timing$(MM).gnuplot: timing.gnuplot
	if test x"$(MM)" != x"" ; then cat timing.gnuplot | sed 's,timing,timing$(MM),' > $@ ; fi

display: timing timing$(MM).gnuplot
	echo "set term x11;" | gnuplot -persist - timing$(MM).gnuplot

timing$(MM).ps:	timing
	echo "set term postscript; set output 'timing$(MM).ps';" \
	  | gnuplot - timing.gnuplot

timing$(MM).ppm: timing
	echo "set term pbm color; set output 'timing$(MM).ppm';" \
          | gnuplot - timing.gnuplot

clean:
	rm -f $(BIN) hw2.dvi hw2.log hw2.aux *.o $(OURS).s
	@cd asm && $(MAKE) clean

realclean: clean
	rm -f *~ timing timing*.raw timing*.ps timing*.ppm
