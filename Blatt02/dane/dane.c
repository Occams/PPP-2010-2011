#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"
#include <stdbool.h>
#include "ppp_pnm.h"
#include "common.h"

#define SCALE_GREYMAP(a,a_min,a_max,n_min,n_max) \
	((( ((a)-(a_min)) * ((n_max) - (n_min)) + (((a_max) - (a_min)) / 2) ) \
	/ \
	((a_max) - (a_min)) ) \
	+ (n_min))

int mpi_self, mpi_processors, a_min = 0, a_max = 0,n_max = (2<<7) - 1, n_min = 0,
*image, maxcolor, rows, cols;
enum pnm_kind kind;

typedef enum {
	SEQUENTIAL,
	OPENMP,
	COMBINED
} Method;

void sequential_determineMinMax();
void sequential_rescale();
void openmp_determineMinMax();
void openmp_determineMinMax_reduction();
void openmp_rescale();

int main(int argc, char **argv) {
	Method method = 0;
	int option;
	char *input_path = "input.pgm", *output_path = "output.pgm";
	
	/* init MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_self);
	
	omp_set_dynamic(false);
	
	/* Read cmdline params */
	while ((option = getopt(argc,argv,"m:x:y:i:o:")) != -1) {
		
		switch(option) {
		case 'm': method = atoi(optarg); break;
		case 'x': n_min = atoi(optarg); break;
		case 'y': n_max = atoi(optarg); break;
		case 'i': input_path = optarg; break;
		case 'o': output_path = optarg; break;
		default:
			return 1;
		}
	}
	
	image = ppp_pnm_read(input_path, &kind, &rows, &cols, &maxcolor);
	
	// printf("Method: %i\n", method);
	// printf("N-Min: %i\n", n_min);
	// printf("N-Max: %i\n", n_max);
	// printf("Input-Path: %s\n", input_path);
	// printf("Output-Path: %s\n", output_path);
	// printf("Rows: %i\n", rows);
	// printf("Cols: %i\n", cols);
	
	/* Validate params */
	if (image == NULL || n_min > n_max || n_min < 0 || n_max >= 2<<7) {
		printf("Invalid params\n");
		return 1;
	}
	
	switch(method) {
	case SEQUENTIAL: sequential_determineMinMax(); sequential_rescale(); break;
	case OPENMP: openmp_determineMinMax(); openmp_rescale(); break;
	case COMBINED: break;
	default: printf("Method not available! Will exit"); return 1;
	}
	
	// if (ppp_pnm_write(output_path, kind, rows, cols, maxcolor, image) != 0) {
	// printf("Write error\n");
	// }
	
	//printf("A-Min: %i\n", a_min);
	//printf("A-Max: %i\n", a_max);
	
	free(image);
	
	return 0;
}

void sequential_determineMinMax() {
	int pixel, x = 0, y = 0;
	double start = seconds();
	a_min = maxcolor;
	
	for (y=0; y<rows; y++) {
		
		for (x=0; x<cols; x++) {
			pixel = image[y*cols+x];
			a_min = MIN(pixel, a_min);
			a_max = MAX(pixel, a_max);
		}
	}
	
	printf("Sequential min/max: %f\n", seconds() - start);
}

void sequential_rescale() {
	int idx, x = 0, y = 0;
	double start = seconds();
	
	for (y=0; y<rows; y++) {
		
		for (x=0; x<cols; x++) {
			idx = y*cols+x;
			image[idx] = SCALE_GREYMAP(image[idx], a_min, a_max, n_min, n_max);
		}
	}
	
	printf("Sequential rescale: %f\n", seconds() - start);
}

void openmp_determineMinMax() {
	int x = 0, y = 0;
	double start = seconds(), elapsed;
	
	a_min = maxcolor;
	
	#pragma omp parallel for private(x)
	for (y=0; y<rows; y++) {
		int pixel, a_min_t = maxcolor, a_max_t = 0;
		
		for (x=0; x<cols; x++) {
			pixel = image[y*cols+x];
			a_min_t = MIN(pixel, a_min_t);
			a_max_t = MAX(pixel, a_max_t );
		}
		
		#pragma omp critical
		{
			a_min = MIN(a_min,a_min_t);
			a_max = MAX(a_max,a_max_t);
		}
	}
	
	elapsed = seconds() - start;
	printf("OPENMP min/max: %f\n", elapsed);
}

void openmp_rescale() {
	int x = 0, y = 0;
	double start = seconds();
	
	#pragma omp parallel for private(x)
	for (y=0; y<rows; y++) {
		
		#pragma omp parallel for
		for (x=0; x<cols; x++) {
			int idx = y*cols+x;
			image[idx] = SCALE_GREYMAP(image[idx], a_min, a_max, n_min, n_max);
		}
	}
	
	printf("OPENMP rescale: %f\n", seconds() - start);
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
