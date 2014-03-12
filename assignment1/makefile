all: rs_pthreads rs_openmp rs_mpi

rs_pthreads:
	gcc -pthread -o rs_pthreads rs_pthreads.c

rs_openmp:
	gcc -o rs_openmp rs_openmp.c

rs_mpi:
	mpicc -o rs_mpi rs_mpi.c

clean:
	rm rs_mpi rs_openmp rs_pthreads
