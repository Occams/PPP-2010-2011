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
	
#define MASTER 0

int mpi_self, mpi_processors, a_min = 0, a_max = 0,n_max = (2<<7) - 1, n_min = 0,
*image, *image_part, image_part_length, maxcolor, rows, cols;
enum pnm_kind kind;

typedef enum {
	SEQUENTIAL,
	OPENMP,
	COMBINED
} Method;

void execute(double (*load) (char *input_path), double (*determineMinMax)(int *image), double (*rescale)(), char * input_path, int count);
double sequential_determineMinMax();
double sequential_rescale();
double sequential_load(char *input_path);
double openmp_determineMinMax(int *image);
double openmp_rescale();
void mpi(char *input_path, int count);
void print(double load, double scatter, double reduction, double min_max, double rescale, double gather);
int * mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length);
void printhelp();

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
			MPI_Finalize();
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
		MPI_Finalize();
		return 1;
	}
	
	switch(method) {
	case SEQUENTIAL: 
		if (mpi_self == MASTER)
		printf(">>SEQUENTIAL\n");
		execute(sequential_load, sequential_determineMinMax, sequential_rescale, input_path, count);
		break;
	case OPENMP:
		if (mpi_self == MASTER)
		printf(">>OPENMP\n");
		execute(sequential_load, openmp_determineMinMax, openmp_rescale, input_path, count);
		break;
	case COMBINED: 
		if (mpi_self == MASTER)
		printf(">>MPI\n");
		mpi(input_path, count); break;
	default: 
		printf("Method not available! Will exit");
		MPI_Finalize();
		return 1;
	}
	
	//if (ppp_pnm_write(output_path, kind, rows, cols, maxcolor, image) != 0) {
	//	printf("Write error\n");
	//}
	
	//printf("A-Min: %i\n", a_min);
	//printf("A-Max: %i\n", a_max);
	
	/* MPI beenden */
	MPI_Finalize();
	return 0;
}

void execute(double (*load) (char *input_path), double (*determineMinMax)(int *image), double (*rescale)(), char * input_path, int count) {
	double load_s = 0, min_max_s = 0, rescale_s = 0; 
	int i;
	
	for (i = 0;  i < count; i++) {
		load_s += load(input_path);
		min_max_s += determineMinMax(image);
		rescale_s += rescale();
		
		if (i +1 < count) {
		free(image);
		}
	}
	
	if (mpi_self == MASTER)
	print(load_s/count, 0, 0, min_max_s / count, rescale_s / count, 0);
}

double sequential_load(char* input_path) {
	double start = seconds();
	image = ppp_pnm_read(input_path, &kind, &rows, &cols, &maxcolor);
	
	if (image == NULL) {
		MPI_Finalize();
		exit(1);
	}
		
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

double openmp_determineMinMax(int* image) {
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

void mpi(char *input_path, int count) {
	int i,x,max_min_buf_s[2], max_min_buf_r[2];
	double load_s = 0, min_max_s = 0, rescale_s = 0,scatter_s = 0, reduction_s = 0, gather_s = 0, start; 
	
	for (i = 0; i < count; i++) {
	
		/* Read one's part of the image */
		start = seconds();
		image_part = ppp_pnm_read_part(input_path, &kind, &rows, &cols, &maxcolor, mpi_read_part);
		load_s += seconds() - start;
		
		if (image_part == NULL) {
			MPI_Finalize();
			exit(1);
		}
			
		//printf("Processors %i owns a part of %i * 4 bytes\n", mpi_self, image_part_length);
		//printf("Total %i * 4 bytes\n",rows*cols);
		
		/* Determine Min/Max */
		double start = seconds();
		a_min = maxcolor;
	
		for (x=0; x < image_part_length; x++) {
			a_min = MIN(image_part[x], a_min);
			a_max = MAX(image_part[x], a_max);
		}
		
		min_max_s += seconds() - start;
		
		/* Allreduce Min/Max */
		max_min_buf_s[0] = a_max;
		max_min_buf_s[1] = -1 * a_min;
		MPI_Allreduce(max_min_buf_s, max_min_buf_r, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		printf("Allreduce Processor %i: Min %i Max %i\n",mpi_self, -1*max_min_buf_r[1],max_min_buf_r[0]);
		
		
		if (i +1 < count) {
			free(image);
		}
	
		free(image_part);
	}
	
	image = image_part;
	
	if (mpi_self == MASTER)
	print(load_s/count, scatter_s/count, reduction_s/count, min_max_s/count, rescale_s/count, gather_s/count);
}

void print(double load, double scatter, double reduction, double min_max, double rescale, double gather) {
	printf("Loading: %f\n", load);
	printf("Min/Max: %f\n", min_max);
	printf("Rescale: %f\n", rescale);
	printf("Scatter: %f\n", scatter);
	printf("Reduction: %f\n", reduction);
	printf("Gather: %f\n", gather);
	printf(">TOTAL: %f\n", gather+load+min_max+rescale+scatter+reduction);
}

int *mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length) {
	
	if (kind != PNM_KIND_PGM) {
		return NULL;
	}
	
	if (mpi_self < (rows*columns) % mpi_processors) {
		*length = (rows*columns) / mpi_processors + 1;
		*offset = *length * mpi_self;
    } else {
		*length = (rows*columns) / mpi_processors;
		*offset = *length * mpi_self  +  (rows*columns) % mpi_processors;
    }
	
	image_part_length = *length;
	
    return (int *) malloc(image_part_length * sizeof(int));
}

void printhelp() {
	if (mpi_self == MASTER) {
		printf("Usage:\n");
		printf("-i 	Input path (DEFAULT: input.pgm)\n");
		printf("-o	Output path (DEFAULT: output.pgm)\n");
		printf("-m	Method: 0=SEQUENTIAL | 1=OPENMP | 2=MPI (DEFAULT: 0)\n");
		printf("-x	N_MIN (DEFAULT: 0)\n");
		printf("-y	N_MAX (DEFAULT: 255)\n");
	}
}
