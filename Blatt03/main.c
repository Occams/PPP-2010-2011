#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>
#include "omp.h"
#include <stdbool.h>
#include "ppp_pnm.h"
#include "common.h"
#include <math.h>
#include <stdarg.h>
#include <sobel.h>
#include <vcd.h>

#define MASTER 0

/* Globals */
int mpi_self = MASTER, mpi_processors;

/* Prototypes */
int m_printf(char *format, ... );
int *mpi_read_part_sobel(enum pnm_kind kind, int rows, int columns, int *offset, int *length);
void printhelp();

int main(int argc, char **argv) {
	int option;
	char *input_path = "input.pgm", *output_path = "output.pgm";
	bool sobel = true, vcd = false, parallel = false;
	int sobel_c = 1;
	
	/* Read cmdline params */
	while ((option = getopt(argc,argv,"s:v:p:i:o:c:h")) != -1) {
	
		switch(option) {
		case 's': sobel = atoi(optarg); break;
		case 'v': vcd = atoi(optarg); break;
		case 'p': parallel = atoi(optarg); break;
		case 'i': input_path = optarg; break;
		case 'o': output_path = optarg; break;
		case 'c': sobel_c = atoi(optarg); break;
		default:
			printhelp();
			return 1;
		}
	}
	
	/* Init MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_self);
	
	if (parallel) {
		if (vcd) {
			
		}
		
		if (sobel) {
			sobel_mpi_init(mpi_self, mpi_processors);
			enum pnm_kind kind;
			int rows, columns, maxcolor;
			int* image = ppp_pnm_read_part(input_path, &kind, &rows, &columns, &maxcolor, sobel_mpi_read_part);
			int dest[rows*columns];
			sobel_parallel(image,dest,rows,columns,sobel_c);
		}
	} else {
	
		if (vcd) {
			enum pnm_kind kind;
			int rows,columns,maxcolor;
			int *image = ppp_pnm_read(input_path, &kind, &rows, &columns, &maxcolor);
			vcd_sequential(image,rows,columns,maxcolor);
			ppp_pnm_write(output_path, kind, rows, columns, maxcolor, image);
		}
		
		if (sobel) {
			enum pnm_kind kind;
			int rows,columns,maxcolor;
			int *image = ppp_pnm_read(input_path, &kind, &rows, &columns, &maxcolor);
			int dest[rows*columns];
			sobel_seq(image,dest,rows,columns,sobel_c);
			ppp_pnm_write(output_path, kind, rows, columns, maxcolor, dest);
		}
	}
	
	return 0;
}



int *mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length) {
	
	if (kind != PNM_KIND_PGM) {
		return NULL;
	}
	
	return (int *) malloc(*length * sizeof(int));
}

void printhelp() {
		m_printf("Usage:\n");
		m_printf("-i  Input path (DEFAULT: input.pgm)\n");
		m_printf("-o  Output path (DEFAULT: output.pgm)\n");
		m_printf("-s  BOOL	Apply Sobel algorithm (DEFAULT: true)\n");
		m_printf("-v  BOOL	Apply VCD algorithm (DEFAULT: false)\n");
		m_printf("-p  BOOL	Parallel execution (DEFAULT: false)\n");
		m_printf("-c  INT	Sobel c parameter (DEFAULT: 1)\n");
		m_printf("-h  This message\n");
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

