.DELETE_ON_ERROR:
.PHONY: all test clean

SUBDIRS:=cs8803ss-project cs7260-project cs4803dgc-project

all:
	for i in $(SUBDIRS) ; do cd $$i && make all && cd - ; done

test:
	for i in $(SUBDIRS) ; do cd $$i && make test && cd - ; done

clean:
	for i in $(SUBDIRS) ; do cd $$i && make clean && cd - ; done
