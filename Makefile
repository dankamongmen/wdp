.DELETE_ON_ERROR:
.PHONY: all clean

SUBDIRS:=cs8803ss-project cs7260-project

all:
	for i in $(SUBDIRS) ; do cd $$i && make && cd - ; done

clean:
	for i in $(SUBDIRS) ; do cd $$i && make clean && cd - ; done
