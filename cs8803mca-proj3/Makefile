#.DELETE_ON_ERROR:
.PHONY: default clean FORCE

SOURCES=router.cpp project2.cc interface.cpp

default: .tags easytest testtraceab testtraceb testtracea report.pdf

easytest: super_easy.txt sim FORCE
	./sim -f ./$< -P 15 -C 10 -B 5 -S 2 -l 64 -v 2 -u 6 > $@
	@diff -ur $@ super_easy_sa.debug | tail

testtraceab: trace_ab.txt sim FORCE
	./sim -f ./$< -P 15 -C 10 -B 5 -S 2 -l 64 -v 2 -u 6 > $@
	@diff -ur $@ trace_ab_l64_v2_u6.final | tail
	./sim -f ./$< -P 15 -C 10 -B 5 -S 2 -l 320 -v 1 -u 1 > $@
	@diff -ur $@ trace_ab_l320_v1_u1.final | tail

testtraceb: trace_b.txt sim FORCE
	./sim -f ./$< -P 15 -C 10 -B 5 -S 2 -l 64 -v 2 -u 6 > $@
	@diff -ur $@ trace_b_l64_v2_u6.final | tail
	./sim -f ./$< -P 15 -C 10 -B 5 -S 2 -l 128 -v 1 -u 3 > $@
	@diff -ur $@ trace_b_l128_v1_u3.final | tail

testtracea: trace_a.txt sim FORCE
	./sim -f ./$< -P 15 -C 10 -B 5 -S 2 -l 64 -v 2 -u 6 > $@
	@diff -ur $@ trace_a_l64_v2_u6.final | tail
	./sim -f ./$< -P 15 -C 10 -B 5 -S 2 -l 32 -v 4 -u 4 > $@
	@diff -ur $@ trace_a_l32_v4_u4.final | tail

sim: $(SOURCES) interface.h router.h cobf.h uncobf.h
	g++ -g -ggdb -O0 -Wall $(SOURCES) -lm -o $@

report.pdf: report.tex
	pdflatex $<

.tags: $(SOURCES) interface.h router.h cobf.h uncobf.h
	ctags -f $@ -R $^
	
clean:
	rm -f sim easytest testtracea testtraceb testtraceab .tags report.pdf report.aux report.log

FORCE:
