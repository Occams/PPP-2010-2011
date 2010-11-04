#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"
#include <stdbool.h>
#include "ppp_pnm.h"

double seconds() {
	struct timeval time;
	gettimeofday(&time, NULL);
	return (double)time.tv_sec + ((double)time.tv_usec)/1000000.0;
}

int *allocints(int size) {
	int *p;
	p = (int *)malloc(size * sizeof(int));
	return p;
}

int mpi_self, mpi_processors;

int main(int argc, char **argv) {
	enum pnm_kind kind;
    int rows, columns, maxcolor;
    int *image;
	char *path = "images/test.pgm";
	
    /*
     * Load the image (name in argv[1]),
     * store the kind (PBM, PGM, PPM) in 'kind',
     * the number of rows and columns in 'rows' and 'columns',
     * the maximal gray value of the image format (NOT the
     * maximal gray value used in the image) in 'maxcolor' and return
     * the image row-wise with one int per pixel.
     */
    image = ppp_pnm_read(argv[1], &kind, &rows, &columns, &maxcolor);
	printf("Rows: %i\n", rows);
	printf("Columns: %i\n", columns);
	printf("Maxcolor: %i\n", maxcolor);
	printf("Kind: %i\n", kind);
	printf("Path: %s", argv[1]);
	ppp_pnm_write(path, kind, rows, columns, maxcolor, image);
	
	/* init MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_self);
	
	return 0;
}

void printhelp() {
	printf("Usage:\n");
	printf("-l 	Size of the integer array to be transmitted (DEFAULT: 2)\n");
	printf("-b	Use native MPI_Bcast\n");
	printf("-i BOOL nonblocking	Use a simulated broadcast (DEFAULT: blocking)\n");
	printf("-t INT branchCount	Use a tree broadcast algorithm with the desired count of branches (DEFAULT: 2)\n");
	printf("-n INT	Iterate the broadcast n times (DEFAULT: 1)\n");
	printf("-s INT	The rank of the source process (DEFAULT: 0)\n");
	printf("-h	This help message\n");
}