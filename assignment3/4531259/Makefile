CC = gcc
PCC = mpicc
OPTFLAGS = -O3 -fno-strict-aliasing -D_GNU_SOURCE
COPTIONS = -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall\
           -Wno-unused-function -Wno-unused-label -Wno-unused-variable\
           -Wno-parentheses -Wsequence-point

#Standard Libraries
STDLIBS = -lm

CFLAGS = $(COPTIONS)  $(OPTFLAGS)
LIBS = $(STDLIBS)

PSOURCE = mt.c getopt.c cmdline.c util.c
PEXE = matvec_mpi

matvec_mpi:
	$(PCC) $(CFLAGS) $(PSOURCE) $(LIBS) -o $(PEXE) -g

