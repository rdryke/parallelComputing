all: rs_pthreads rs_openmp rs_mpi

CS: countingSort.c countingSortSimple.c
	gcc -o CS countingSort.c
	gcc -o CSS countingSortSimple.c

rs_pthreads: rs_pthreads.c
	gcc -pthread -o rs_pthreads rs_pthreads.c

rs_openmp: rs_openmp.c
	gcc -fopenmp -g -o rs_openmp rs_openmp.c

rs_mpi: rs_mpi.c
	mpicc -o rs_mpi rs_mpi.c

clean:
	rm rs_mpi rs_openmp rs_pthreads CS CSS
