# ====================
# Compiler settings
#
# You may want to edit these based on whatever compilers you have
# available to you.

# --- GNU compiler ---
#export CC = gcc-4.4
export CC?=gcc43
#F77 = gfortran-4.3
F77 = gfortran43
O2FLAGS:=-O2 -fomit-frame-pointer -funroll-loops -finline-functions
OFLAGS:=$(O2FLAGS) -fno-inline -fmodulo-sched -fipa-pta -combine \
	-ftree-loop-linear -ftree-loop-im
CFLAGS:=-std=gnu99 -pipe -march=native -Wall -W -Wextra -Werror $(OFLAGS) \
	-ffast-math -mfpmath=sse -funsafe-math-optimizations -D_GNU_SOURCE 
#	-fwhole-program

#LIBS = -lm

NB=64

# --- Intel compiler for Core2 Quad systems ---
#CC = /opt/intel/Compiler/11.0/084/bin/intel64/icc
#F77 = ifort
#CFLAGS:=-B /opt/intel/Compiler/11.0/084/bin/intel64 -restrict -std=gnu99 -O3 -xHost -ip -ipo -gcc -vec-report3 -ftz -opt-ra-region-strategy=block
#LIBS = -lm

ifeq "$(shell uname -s)" "Linux"
  LIBS += -lrt
endif

# ====================
# BLAS settings (optional)
#
# Set the variable, 'BLASLIB', to the appropriate linker flags.
#

BLASLIB = -Wl,-R/nethome/nblack6/lib -L/nethome/nblack6/lib -lblas

# --- AMD / ACML BLAS ---
#ACML_LIBDIR = /usr/local/apps/acml-blas/4.1.0/gfortran64/lib
#BLASLIB = -L$(ACML_LIBDIR) -Wl,-rpath -Wl,$(ACML_LIBDIR) -lacml

# --- Intel BLAS (MKL 10.x) ---
#MKL_LIBDIR = /opt/intel/Compiler/11.1/046/mkl/lib/em64t
#BLASLIB = -L$(MKL_LIBDIR) -Wl,-rpath -Wl,$(MKL_LIBDIR) -lmkl_blas95

ifeq ($(CC),icc)
  ifeq "$(shell uname -s)" "Darwin"
    BLASDIR = /Library/Frameworks/Intel_MKL.framework/Libraries/32
    BLASLIB += -L$(BLASDIR) -Wl,-rpath -Wl,$(BLASDIR)
    BLASLIB += -lmkl_intel
    BLASLIB += -lmkl_sequential
    BLASLIB += -lmkl_core
    ICCDIR = /opt/intel/Compiler/11.0/054/lib
    BLASLIB += -L$(ICCDIR) -Wl,-rpath -Wl,$(ICCDIR) -lirc
  endif
endif

# eof
