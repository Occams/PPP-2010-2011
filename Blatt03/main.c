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
#include <pgm_distribute.h>

#define MASTER 0

/* Globals */
int mpi_self = MASTER, mpi_processors;

/* Prototypes */
int m_printf(char *format, ... );
int *mpi_read_part_sobel(enum pnm_kind kind, int rows, int columns, int *offset, int *length);
void printhelp();

int main(int argc, char **argv) {
	int x, option, *gath_image = NULL, *gath_counts = NULL, *gath_displs = NULL,
	sobel_c = 1, rows,columns,maxcolor,*image = NULL;
	char *input_path = "input.pgm", *output_path = "output.pgm";
	bool sobel = true, vcd = false, parallel = false;
	enum pnm_kind kind;
	pgm_part mypart, info;
	
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
		if(vcd) vcd_mpi_init(mpi_self, mpi_processors);
		if(sobel) sobel_mpi_init(mpi_self, mpi_processors);
		
		pgm_distribute_init(mpi_processors);
		
		if(vcd) image = ppp_pnm_read_part(input_path, &kind, &rows, &columns, &maxcolor, vcd_mpi_read_part);
		if(sobel) image = ppp_pnm_read_part(input_path, &kind, &rows, &columns, &maxcolor, sobel_mpi_read_part);
		
		pgm_partinfo(rows, mpi_self, &mypart);
		
		if(vcd) vcd_parallel(image, mypart.rows, columns, maxcolor);
		if(sobel) sobel_parallel(image, mypart.rows, columns,sobel_c, maxcolor);
		
		
		if(mpi_self == MASTER) gath_image = (int*)malloc(sizeof(int)*rows*columns);
		gath_counts = (int*)malloc(sizeof(int)*mpi_processors);
		gath_displs = (int*)malloc(sizeof(int)*mpi_processors);
		

		if( (mpi_self == MASTER && gath_image == NULL) || gath_counts == NULL || gath_displs == NULL) {
			printf("Alloc failed... Will exit now...\n");
			MPI_Finalize();
			return 1;
		}
		
		
		for(x = 0; x < mpi_processors; x++) {
			pgm_partinfo(rows, x, &info);
			gath_counts[x] = columns*(info.rows - info.overlapping_top - info.overlapping_bot);
			gath_displs[x] = columns*(info.offset + info.overlapping_top);
		}
		
		MPI_Gatherv(
		image+mypart.overlapping_top*columns,
		gath_counts[mpi_self],
		MPI_INT,
		gath_image, gath_counts, gath_displs, MPI_INT, MASTER, MPI_COMM_WORLD);
		
		if(mpi_self == MASTER) {
			ppp_pnm_write(output_path, kind, rows, columns, maxcolor, gath_image);
		}
		
		free(gath_image);
		free(gath_counts);
		free(gath_displs);
	} else {
		image = ppp_pnm_read(input_path, &kind, &rows, &columns, &maxcolor);
		
		if (vcd) {
			vcd_sequential(image,rows,columns,maxcolor);
		}
		
		if (sobel) {
			sobel_seq(image,rows,columns,sobel_c, maxcolor);
		}
		
		ppp_pnm_write(output_path, kind, rows, columns, maxcolor, image);
		free(image);
	}
	
	MPI_Finalize();
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

