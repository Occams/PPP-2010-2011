#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "ppp_pnm.h"
#include <mpi.h>
#include "omp.h"
#include <common.h>

#define GV_SCALE(a, amin, amax, nmin, nmax) \
		(( \
			( (a) - (amin) ) * ( (nmax) - (nmin) ) + ( (amax) - (amin) ) / 2 \
		) \
		/ ( (amax) - (amin) ) + (nmin))

		
/*
 * Typedefs...
 */
typedef enum {
	CONV_SEQUENTIAL,
	CONV_OPENMP,
	CONV_MPI
} ConvOption;

/*
 * Declarations
 */
void help(void);
 
void sequential_scale(int *image, int rows, int columns, int maxcolor,
	int amin, int amax, int nmin, int nmax);
void sequential_min_max_grayvals(int* image, int rows, int columns, int maxcolor, int* min, int* max);

void openmp_scale(int *image, int rows, int columns, int maxcolor,
	int amin, int amax, int nmin, int nmax);
void openmp_min_max_grayvals(int* image, int rows, int columns, int maxcolor, int* min, int* max);

int mpi_np;
int mpi_self;

/* Time measurements */
double t_load, t_scatter, t_reduction, t_scale, t_gather;

/*
 * Implementations
 */
int main(int argc, char *argv[]) {
	/*
	 * User input options...
	 */
	ConvOption conv;
	int nmin, nmax;
	nmin = 0; nmax = 255;
	char *input, *output;
	
	
	int option;
    while ((option = getopt(argc,argv,"i:o:a:b:c:")) != -1) {
        switch(option) {
        case 'i': input = optarg; break;
        case 'o': output = optarg; break;
        case 'a': nmin = atoi(optarg); break;
        case 'b': nmax = atoi(optarg); break;
        case 'c': conv = atoi(optarg); break;
        case 'h': help(); return 0; break;
        default:
            return 1;
        }
    }
    
	/*
	 * Open image and read out the amin and amax values.
	 */
	enum pnm_kind kind;
	int rows, columns, maxcolor;
	int* image;
	
	int amin, amax;
	switch(conv) {
		case CONV_SEQUENTIAL:
			printf("Convert Sequential...\n");
			image = ppp_pnm_read(input, &kind, &rows, &columns, &maxcolor);
			sequential_min_max_grayvals(image, rows, columns, maxcolor, &amin, &amax);
			sequential_scale(image, rows, columns, maxcolor,
				amin, amax, nmin, nmax);
			ppp_pnm_write(output, kind, rows, columns, maxcolor, image);
		break;
		
		case CONV_OPENMP:
			printf("Convert Openmp...\n");
			image = ppp_pnm_read(input, &kind, &rows, &columns, &maxcolor);
			openmp_min_max_grayvals(image, rows, columns, maxcolor, &amin, &amax);
			openmp_scale(image, rows, columns, maxcolor,
				amin, amax, nmin, nmax);
			ppp_pnm_write(output, kind, rows, columns, maxcolor, image);
		break;
		
		case CONV_MPI: {
			MPI_Init(&argc, &argv);
			MPI_Comm_size(MPI_COMM_WORLD, &mpi_np);
			MPI_Comm_rank(MPI_COMM_WORLD, &mpi_self);
			int i; // Common iteration variable
			
			/*
			 * Read the image if current is the master...
			 */
			if(mpi_self == 0) {
				t_load = seconds();
				image = ppp_pnm_read(input, &kind, &rows, &columns, &maxcolor);
				t_load = seconds() - t_load;
			}
			
			/*
			 * Broadcast the size of the image, as well as the maxcolor
			 */
			int image_meta[] = {rows, columns, maxcolor};
			int counts[mpi_np], displs[mpi_np];
			MPI_Bcast(image_meta, 3, MPI_INT, 0, MPI_COMM_WORLD);
			
			
			/*
			 * Prepare parts for scatterv
			 */
			int partSize = (image_meta[0]*image_meta[1])/mpi_np;
			for(i = 0; i < mpi_np; i++) {
				counts[i] = partSize;
				displs[i] = partSize*i;
			}
			counts[mpi_np-1] = (image_meta[0]*image_meta[1])%partSize;
			counts[mpi_np-1] = counts[mpi_np-1]==0?partSize:counts[mpi_np-1];
			
			/*
			 * Allocate space for the image on receiving side...
			 */
			if(mpi_self != 0) {
				image = (int*)malloc(partSize*sizeof(int));
			}
			
			/*
			 * Scatter data...
			 */
			MPI_Barrier(MPI_COMM_WORLD);
			t_scatter = seconds();
			MPI_Scatterv(image, counts, displs, MPI_INT, image, counts[mpi_self], MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			t_scatter = seconds() - t_scatter;
			
			/*
			 * Calculate max...
			 */
			int max, min;
			max = 0;
			min = maxcolor;
			for(i = 0; i < counts[mpi_self]; i++) {
				max = MAX(max, image[i]);
				min = MIN(min, image[i]);
			}
			
			int g_max, g_min;
			MPI_Barrier(MPI_COMM_WORLD);
			t_reduction = seconds();
			MPI_Allreduce(&max, &g_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
			MPI_Allreduce(&min, &g_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			t_reduction = seconds() - t_reduction;

			/*
			 * Now, as everybody has the min and max, scale the values locally
			 */
			t_scale = seconds();
			for (i = 0; i < counts[mpi_self]; i++) {
				image[i] = GV_SCALE(image[i], g_min, g_max, nmin, nmax);
			}
			t_scale = seconds() - t_scale;
			
			/*
			 * Gather the image again...
			 */
			MPI_Barrier(MPI_COMM_WORLD);
			t_gather = seconds();
			MPI_Gatherv(image, counts[mpi_self], MPI_INT, image, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			t_gather = seconds() - t_gather;
			
			if(mpi_self == 0) {
				ppp_pnm_write(output, kind, rows, columns, maxcolor, image);
			}
			
			/* MPI beenden */
			MPI_Finalize();
			
			if(mpi_self == 0) {
				printf("Load: %f\nScatter: %f\nReduction: %f\nScale: %f\nGather: %f\n",
					t_load, t_scatter, t_reduction, t_scale, t_gather);
			}
		break;}
		
		default:
		break;
	}
	
	if(image != NULL) {
	    free(image);
	}
	
	return 0;
}



/*
 * Sequential implementations...
 */
void sequential_scale(int *image, int rows, int columns, int maxcolor,
	int amin, int amax, int nmin, int nmax) {
	
	int x, y;
	for (y=0; y<rows; y++) {
		for (x=0; x<columns; x++) {
			image[y*columns+x] = GV_SCALE(image[y*columns+x], amin, amax, nmin, nmax);
		}
	}
}

void sequential_min_max_grayvals(int* image, int rows, int columns, int maxcolor, int* min, int* max) {
	int amin, amax;
	
	amin = maxcolor;
	amax = 0;
	
	int x, y;
	for (y=0; y<rows; y++) {
		for (x=0; x<columns; x++) {
			int color = image[y*columns+x];
			amin = MIN(amin,color);
			amax = MAX(amax,color);
		}
	}
	
	*min = amin;
	*max = amax;
}




/*
 * Openmp implementations...
 */
void openmp_scale(int *image, int rows, int columns, int maxcolor,
	int amin, int amax, int nmin, int nmax) {
	
	int x, y;
#pragma omp parallel for private(x)
	for (y=0; y<rows; y++) {
		for (x=0; x<columns; x++) {
			image[y*columns+x] = GV_SCALE(image[y*columns+x], amin, amax, nmin, nmax);
		}
	}
}

void openmp_min_max_grayvals(int* image, int rows, int columns, int maxcolor, int* min, int* max) {
	int amin, amax;
	
	amin = maxcolor;
	amax = 0;
	
	int x, y;
#pragma omp parallel for private(x)
	for (y=0; y<rows; y++) {
		for (x=0; x<columns; x++) {
			int color = image[y*columns+x];
			#pragma omp critical
			{
				amin = MIN(amin,color);
				amax = MAX(amax,color);
			}
		}
	}
	
	*min = amin;
	*max = amax;
}


void help() {
	printf("Usage:\n");
	printf("-c	Convert Type: 0 Sequential, 1 Openmp, 2 MPI \n");
	printf("-a	Integer for nmin\n");
	printf("-b	Integer for nmax\n");
	printf("-i	Path to input file\n");
	printf("-o	Path to output file\n");
}
