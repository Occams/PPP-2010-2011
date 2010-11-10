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

void execute(double load(char *input_path), double determineMinMax(), double rescale(),
	void print(double load, double min_max, double scale), char * input_path, int count);
double sequential_determineMinMax();
double sequential_rescale();
double sequential_load(char* input_path);
double openmp_determineMinMax();
double openmp_rescale();
double mpi_determineMinMax();
double mpi_rescale();
double mpi_load(char* input_path);
void sequential_print(double load, double min_max, double rescale);
void openmp_print(double load, double min_max, double rescale);
void mpi_print(double load, double scatter, double reduction, double min_max, double rescale, double gather);
int * mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length);

int main(int argc, char **argv) {
	Method method = 0;
	int option,count = 1;
	char *input_path = "input.pgm", *output_path = "output.pgm";
	
	/* init MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_self);
	
	/* Read cmdline params */
	while ((option = getopt(argc,argv,"m:x:y:i:o:n:")) != -1) {
		
		switch(option) {
		case 'm': method = atoi(optarg); break;
		case 'x': n_min = atoi(optarg); break;
		case 'y': n_max = atoi(optarg); break;
		case 'i': input_path = optarg; break;
		case 'o': output_path = optarg; break;
		case 'n': count = atoi(optarg); break;
		default:
			return 1;
		}
	}
	
	// printf("Method: %i\n", method);
	// printf("N-Min: %i\n", n_min);
	// printf("N-Max: %i\n", n_max);
	// printf("Input-Path: %s\n", input_path);
	// printf("Output-Path: %s\n", output_path);
	// printf("Rows: %i\n", rows);
	// printf("Cols: %i\n", cols);
	
	/* Validate params */
	if (n_min > n_max || n_min < 0 || n_max >= 2<<7 || count < 1) {
		printf("Invalid params\n");
		return 1;
	}
	
	switch(method) {
	case SEQUENTIAL: 
		execute(sequential_load, sequential_determineMinMax, sequential_rescale, sequential_print, input_path,count);
		break;
	case OPENMP:
		image = ppp_pnm_read(input_path, &kind, &rows, &cols, &maxcolor);
		openmp_determineMinMax();
		openmp_rescale();
		break;
	case COMBINED: mpi_determineMinMax(); mpi_rescale(); break;
	default: printf("Method not available! Will exit"); return 1;
	}
	
	if (ppp_pnm_write(output_path, kind, rows, cols, maxcolor, image) != 0) {
		printf("Write error\n");
	}
	
	//printf("A-Min: %i\n", a_min);
	//printf("A-Max: %i\n", a_max);
	
	return 0;
}

void execute(double load(char *input_path), double determineMinMax(), double rescale(),
	void print(double load, double min_max, double scale), char* input_path, int count) {
	double load_s = 0, min_max_s = 0, scale_s = 0; 
	int i;
	
	for (i = 0;  i < count; i++) {
		load_s += load(input_path);
		min_max_s += determineMinMax();
		scale_s += rescale();
		free(image);
	}
	
	print(load_s/count, min_max_s / count, scale_s / count);
}

double sequential_load(char* input_path) {
	double start = seconds();
	image = ppp_pnm_read(input_path, &kind, &rows, &cols, &maxcolor);
	return seconds() - start;
}

double sequential_determineMinMax() {
	int pixel, x = 0, y = 0;
	double start = seconds();
	a_min = maxcolor;
	
	for (y=0; y<rows; y++) {
		
		for (x=0; x<cols; x++) {
			pixel = image[y*cols+x];
			a_min = MIN(pixel, a_min);
			a_max = MAX(pixel, a_max);
		}
		
		if (a_min == 0 && a_max == maxcolor) {
			break;
		}
	}
	
	return seconds() - start;
}

double sequential_rescale() {
	int idx, x = 0, y = 0, rows_l = rows, cols_l = cols,
		a_min_l = a_min, a_max_l = a_max, n_min_l = n_min, n_max_l = n_max;
	double start = seconds();
	
	for (y=0; y<rows_l; y++) {
		
		for (x=0; x<cols_l; x++) {
			idx = y*cols_l+x;
			image[idx] = SCALE_GREYMAP(image[idx], a_min_l, a_max_l, n_min_l, n_max_l);
		}
	}
	
	return seconds() - start;
}

double openmp_determineMinMax() {
	int x = 0, y = 0;
	double start = seconds();
	
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
	
	return seconds() - start;
}

double openmp_rescale() {
	int x = 0, y = 0, rows_l = rows, cols_l = cols,
		a_min_l = a_min, a_max_l = a_max, n_min_l = n_min, n_max_l = n_max;
	double start = seconds();
	
	#pragma omp parallel for private(x)
	for (y=0; y<rows_l; y++) {
		int idx;
		
		#pragma omp parallel for private(idx)
		for (x=0; x<cols_l; x++) {
			idx = y*cols_l+x;
			image[idx] = SCALE_GREYMAP(image[idx], a_min_l, a_max_l, n_min_l, n_max_l);
		}
	}
	
	return seconds() - start;
}

double mpi_determineMinMax() {
	return 0;
}

double mpi_rescale() {
	return 0;
}

void sequential_print(double load, double min_max, double rescale) {
	printf(">>SEQUENTIAL\n");
	printf("Loading: %f\n", load);
	printf("Min/Max: %f\n", min_max);
	printf("Rescale: %f\n", rescale);
}

void omp_print(double load, double min_max, double rescale) {
	printf(">>OPENMP\n");
	printf("Loading: %f\n", load);
	printf("Min/Max: %f\n", min_max);
	printf("Rescale: %f\n", rescale);
}
void mpi_print(double load, double scatter, double reduction, double min_max, double rescale, double gather) {
	printf(">>MPI\n");
	printf("Loading: %f\n", load);
	printf("Min/Max: %f\n", min_max);
	printf("Rescale: %f\n", rescale);
}

int * mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length) {
	*offset = (rows / mpi_processors) * mpi_self * columns;
	*length = (rows / mpi_processors) * mpi_self * columns;
	return (int *) malloc(0);
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
