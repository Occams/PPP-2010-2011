#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"
#include <stdbool.h>
#include "ppp_pnm.h"
#include "common.h"
#include <stdarg.h>

#define SCALE_GREYMAP(a,a_min,a_max,n_min,n_max) \
	((( ((a)-(a_min)) * ((n_max) - (n_min)) + (((a_max) - (a_min)) / 2) ) \
	/ \
	((a_max) - (a_min)) ) \
	+ (n_min))

#define MASTER 0

int mpi_self, mpi_processors, a_min = 0, a_max = 0,n_max = (2<<7) - 1, n_min = 0,
*image, maxcolor, rows, cols;
enum pnm_kind kind;

typedef enum {
	SEQUENTIAL,
	OPENMP,
	COMBINED
} Method;

int m_printf(char *format, ... );
void execute(double (*determineMinMax)(), double (*rescale)(),
	char * input_path, int count);
double sequential_determineMinMax();
double sequential_rescale();
double sequential_load(char *input_path);
double openmp_determineMinMax();
double openmp_rescale();
void mpi(char *input_path, int count);
void print(double load, double reduction, double min_max, double rescale, double gather);
int *mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length);
void printhelp();

int main(int argc, char **argv) {
	Method method = 0;
	int option, count = 1;
	char *input_path = "input.pgm", *output_path = "output.pgm";
	
	/* Init MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_self);
	
	/* Read cmdline params */
	while ((option = getopt(argc,argv,"m:x:y:i:o:n:h")) != -1) {
		
		switch(option) {
		case 'm': method = atoi(optarg); break;
		case 'x': n_min = atoi(optarg); break;
		case 'y': n_max = atoi(optarg); break;
		case 'i': input_path = optarg; break;
		case 'o': output_path = optarg; break;
		case 'n': count = atoi(optarg); break;
		default:
			printhelp();
			MPI_Finalize();
			return 1;
		}
	}
	
	/* Validate params */
	if (n_min > n_max || n_min < 0 || n_max >= 2<<7 || count < 1) {
		m_printf("Invalid params\n");
		MPI_Finalize();
		return 1;
	}
	
	switch(method) {
		case SEQUENTIAL: 
			m_printf("--SEQUENTIAL--\n");
			execute(sequential_determineMinMax, sequential_rescale, input_path, count);
			break;
		case OPENMP:
			m_printf("--OPENMP--\n");
			execute(openmp_determineMinMax, openmp_rescale, input_path, count);
			break;
		case COMBINED: 
			m_printf("--MPI--\n");
			mpi(input_path, count);
			break;
		default: 
			m_printf("Method not available! Will exit now.");
			MPI_Finalize();
			return 1;
	}
	
	/* Write output image */
	if (mpi_self == MASTER) {
		if (ppp_pnm_write(output_path, kind, rows, cols, maxcolor, image) != 0) {
			m_printf("Write error\n");
		}
	}
	
	MPI_Finalize();
	return 0;
}

void execute(double (*determineMinMax)(), double (*rescale)(), char * input_path, int count) {
	double load_s = 0, min_max_s = 0, rescale_s = 0; 
	int i;
	
	for (i = 0;  i < count; i++) {
		load_s += sequential_load(input_path);
		min_max_s += determineMinMax();
		rescale_s += rescale();
		
		/* Do not free image on last iteration. */
		if (i +1 < count) {
			free(image);
		}
	}
	
	print(load_s/count, 0, min_max_s / count, rescale_s / count, 0);
}

double sequential_load(char* input_path) {
	double start = seconds();
	image = ppp_pnm_read(input_path, &kind, &rows, &cols, &maxcolor);
	
	if (image == NULL || kind != PNM_KIND_PGM) {
		m_printf("Error while reading image");
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
		
		/* Minimum and maximum reached. Nothing more to do. */
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
	
	for (x=0; x<rows_l; x++) {
		
		for (y=0; y<cols_l; y++) {
			idx = x*cols_l+y;
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
		
		#pragma omp parallel for
		for (x=0; x<cols_l; x++) {
			int idx = y*cols_l+x;
			image[idx] = SCALE_GREYMAP(image[idx], a_min_l, a_max_l, n_min_l, n_max_l);
		}
	}
	
	return seconds() - start;
}

void mpi(char *input_path, int count) {
	int i, x, max_min_buf_s[2], max_min_buf_r[2], *image_part, 
		recvcounts[mpi_processors], displs[mpi_processors], tmp,
		a_min_l, a_max_l, n_min_l, n_max_l;
	double load_s = 0, min_max_s = 0, rescale_s = 0, reduction_s = 0, gather_s = 0, start; 
	
	for (i = 0; i < count; i++) {
		
		/* Read one's part of the image */
		MPI_Barrier(MPI_COMM_WORLD);
		start = seconds();
		image_part = ppp_pnm_read_part(input_path, &kind, &rows, &cols, &maxcolor, mpi_read_part);
		MPI_Barrier(MPI_COMM_WORLD);
		load_s += seconds() - start;
		
		tmp = (rows*cols) % mpi_processors;
		
		/* Initialize length and offset arrays */
		for (x = 0; x < mpi_processors; x++) {
			recvcounts[x] = x < tmp ? (rows*cols) / mpi_processors + 1 : (rows*cols) / mpi_processors;
			displs[x] = x < tmp ? x * recvcounts[x] : x * recvcounts[x] +  tmp;
		}
		
		if (image_part == NULL || kind != PNM_KIND_PGM) {
			m_printf("Error while reading image.");
			MPI_Finalize();
			exit(1);
		}
		
		/* Determine minimum and maximum */
		MPI_Barrier(MPI_COMM_WORLD);
		start = seconds();
		a_min = maxcolor;
		
		#pragma omp parallel
		{
			int a_min_t = maxcolor, a_max_t = 0;
			
			#pragma omp for
			for (x=0; x < recvcounts[mpi_self]; x++) {
				a_min_t = MIN(image_part[x], a_min_t);
				a_max_t = MAX(image_part[x], a_max_t);
			}
			
			#pragma omp critical
			{
				a_min = MIN(a_min,a_min_t);
			}
			
			#pragma omp critical
			{
				a_max = MAX(a_max, a_max_t);
			}
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		min_max_s += seconds() - start;
		
		/* Allreduce minima and maxima */
		MPI_Barrier(MPI_COMM_WORLD);
		start = seconds();
		max_min_buf_s[0] = a_max;
		max_min_buf_s[1] = -1 * a_min;
		MPI_Allreduce(max_min_buf_s, max_min_buf_r, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		a_max = max_min_buf_r[0];
		a_min = -1 * max_min_buf_r[1];
		MPI_Barrier(MPI_COMM_WORLD);
		reduction_s += seconds() - start;
		
		/* Rescale one's part of the image */
		a_min_l = a_min, a_max_l = a_max, n_min_l = n_min, n_max_l = n_max;
		MPI_Barrier(MPI_COMM_WORLD);
		start = seconds();
		
		#pragma omp parallel for
		for (x=0; x < recvcounts[mpi_self]; x++) {
			image_part[x] = SCALE_GREYMAP(image_part[x], a_min_l, a_max_l, n_min_l, n_max_l);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		rescale_s += seconds() - start;
		
		/* Gather image data at root process */
		if (mpi_self == MASTER) {
			image = (int *) malloc(rows * cols * sizeof(int));
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		start = seconds();
		MPI_Gatherv(image_part, recvcounts[mpi_self], MPI_INT, image,
			recvcounts, displs, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		gather_s += seconds() - start;
		
		free(image_part);
		
		/* Do not free image on last iteration */
		if (i + 1 < count && mpi_self == MASTER) {
			free(image);
		}
	}
	
	print(load_s/count, reduction_s/count, min_max_s/count, rescale_s/count, gather_s/count);
}

void print(double load, double reduction, double min_max, double rescale, double gather) {
	m_printf("Loading: %f\n", load);
	m_printf("Min/Max: %f\n", min_max);
	m_printf("Rescale: %f\n", rescale);
	m_printf("Reduction: %f\n", reduction);
	m_printf("Gather: %f\n", gather);
	m_printf("->TOTAL: %f\n", gather+load+min_max+rescale+reduction);
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
	
	return (int *) malloc(*length * sizeof(int));
}

void printhelp() {
		m_printf("Usage:\n");
		m_printf("-i 	Input path (DEFAULT: input.pgm)\n");
		m_printf("-o	Output path (DEFAULT: output.pgm)\n");
		m_printf("-m	Method: 0=SEQUENTIAL | 1=OPENMP | 2=MPI (DEFAULT: 0)\n");
		m_printf("-x	N_MIN (DEFAULT: 0)\n");
		m_printf("-y	N_MAX (DEFAULT: 255)\n");
		m_printf("-h	This message\n");
}

int m_printf(char *format, ...) {
	
	if (mpi_self == MASTER) {
		va_list args;
		va_start(args, format);
		int r = vprintf(format, args);
		va_end(args);
		return r;
	} else {
		return -1;
	}
}

