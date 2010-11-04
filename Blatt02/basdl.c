#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "ppp_pnm.h"
#include <mpi.h>
#include "omp.h"

#define GV_SCALE(a, amin, amax, nmin, nmax) \
		(( \
			( (a) - (amin) ) * ( (nmax) - (nmin) ) + ( (amax) - (amin) ) / 2 \
		) \
		/ ( (amax) - (amin) ) + (nmin))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) < (Y) ? (Y) : (X))

		
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
void sequential_scale(int *image, int rows, int columns, int maxcolor,
	int amin, int amax, int nmin, int nmax);
void sequential_min_max_grayvals(int* image, int rows, int columns, int maxcolor, int* min, int* max);

void openmp_scale(int *image, int rows, int columns, int maxcolor,
	int amin, int amax, int nmin, int nmax);
void openmp_min_max_grayvals(int* image, int rows, int columns, int maxcolor, int* min, int* max);

void mpi_scale(int *image, int rows, int columns, int maxcolor,
	int amin, int amax, int nmin, int nmax);
void mpi_min_max_grayvals(int* image, int rows, int columns, int maxcolor, int* min, int* max);
int mpi_np;
int mpi_self;

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
			int partSize = 0;
			
			/*
			 * Scatter...
			 */
			if(mpi_self == 0) {
				image = ppp_pnm_read(input, &kind, &rows, &columns, &maxcolor);
				partSize = (rows*columns)/mpi_np; 
			}
			
			/*
			 * Broadcast the size and allocate space...
			 */
			MPI_Bcast(&partSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if(mpi_self != 0) {
				image = (int*)malloc(partSize*sizeof(int));
			}
			
			/*
			 * Scatter data...
			 */
			MPI_Scatter(image, partSize, MPI_INT, image, partSize, MPI_INT, 0, MPI_COMM_WORLD);
			
			/*
			 * Calculate max...
			 */
			int i;
			int max, min;
			max = 0;
			min = 255;
			for(i = 0; i < partSize; i++) {
				max = MAX(max, image[i]);
				min = MIN(min, image[i]);
			}
			
			int g_max, g_min;
			MPI_Allreduce(&max, &g_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
			MPI_Allreduce(&min, &g_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

			/*
			 * Now, as everybody has the min and max, scale the values locally
			 */
			for (i = 0; i < partSize; i++) {
				image[i] = GV_SCALE(image[i], g_min, g_max, nmin, nmax);
			}
			
			/*
			 * Gather the image again...
			 */
			MPI_Gather(image, partSize, MPI_INT, image, partSize, MPI_INT, 0, MPI_COMM_WORLD);
			
			if(mpi_self == 0) {
				ppp_pnm_write(output, kind, rows, columns, maxcolor, image);
			}
			
			/* MPI beenden */
			MPI_Finalize();
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



/*
 * MPI implementations...
 */
void mpi_scale(int *image, int rows, int columns, int maxcolor,
	int amin, int amax, int nmin, int nmax) {
	
	int x, y;
	for (y=0; y<rows; y++) {
		for (x=0; x<columns; x++) {
			image[y*columns+x] = GV_SCALE(image[y*columns+x], amin, amax, nmin, nmax);
		}
	}
}

void mpi_min_max_grayvals(int* image, int rows, int columns, int maxcolor, int* min, int* max) {
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
		
		/*
		 * If the minimum value is 0 and the maximum color is maxcolor return...
		 */
		if(amin == 0 && amax == maxcolor) {
			y = rows;
		}
	}
	
	*min = amin;
	*max = amax;
}
