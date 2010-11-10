#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "ppp_pnm.h"
#include <mpi.h>
#include "omp.h"
#include <common.h>

#define MEASURES 10
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
double t_load[MEASURES], t_scatter[MEASURES], t_reduction[MEASURES], t_scale[MEASURES], t_gather[MEASURES];

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
	
	int amin, amax, measurepos;	
	for(measurepos = 0; measurepos < MEASURES; measurepos++) {
	
		switch(conv) {
			case CONV_SEQUENTIAL:
				printf("Convert Sequential...\n");
				t_scatter[measurepos] = t_gather[measurepos] = 0.0;
			
				t_load[measurepos] = seconds();
				image = ppp_pnm_read(input, &kind, &rows, &columns, &maxcolor);
				t_load[measurepos] = seconds() - t_load[measurepos];
			
				t_reduction[measurepos] = seconds();
				sequential_min_max_grayvals(image, rows, columns, maxcolor, &amin, &amax);
				t_reduction[measurepos] = seconds() - t_reduction[measurepos];
			
				t_scale[measurepos] = seconds();
				sequential_scale(image, rows, columns, maxcolor,
					amin, amax, nmin, nmax);
				t_scale[measurepos] = seconds() - t_scale[measurepos];
			
				ppp_pnm_write(output, kind, rows, columns, maxcolor, image);
			break;
		
			case CONV_OPENMP:
				printf("Convert Openmp...\n");
				t_scatter[measurepos] = t_gather[measurepos] = 0.0;
			
				t_load[measurepos] = seconds();
				image = ppp_pnm_read(input, &kind, &rows, &columns, &maxcolor);
				t_load[measurepos] = seconds() - t_load[measurepos];
			
				t_reduction[measurepos] = seconds();
				openmp_min_max_grayvals(image, rows, columns, maxcolor, &amin, &amax);
				t_reduction[measurepos] = seconds() - t_reduction[measurepos];
			
			
				t_scale[measurepos] = seconds();
				openmp_scale(image, rows, columns, maxcolor,
					amin, amax, nmin, nmax);
				t_scale[measurepos] = seconds() - t_scale[measurepos];
			
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
					t_load[measurepos] = seconds();
					image = ppp_pnm_read(input, &kind, &rows, &columns, &maxcolor);
					t_load[measurepos] = seconds() - t_load[measurepos];
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
				t_scatter[measurepos] = seconds();
				MPI_Scatterv(image, counts, displs, MPI_INT, image, counts[mpi_self], MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Barrier(MPI_COMM_WORLD);
				t_scatter[measurepos] = seconds() - t_scatter[measurepos];
			
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
				t_reduction[measurepos] = seconds();
				MPI_Allreduce(&max, &g_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
				MPI_Allreduce(&min, &g_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
				MPI_Barrier(MPI_COMM_WORLD);
				t_reduction[measurepos] = seconds() - t_reduction[measurepos];

				/*
				 * Now, as everybody has the min and max, scale the values locally
				 */
				t_scale[measurepos] = seconds();
				for (i = 0; i < counts[mpi_self]; i++) {
					image[i] = GV_SCALE(image[i], g_min, g_max, nmin, nmax);
				}
				t_scale[measurepos] = seconds() - t_scale[measurepos];
			
				/*
				 * Gather the image again...
				 */
				MPI_Barrier(MPI_COMM_WORLD);
				t_gather[measurepos] = seconds();
				MPI_Gatherv(image, counts[mpi_self], MPI_INT, image, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Barrier(MPI_COMM_WORLD);
				t_gather[measurepos] = seconds() - t_gather[measurepos];
			
				if(mpi_self == 0) {
					ppp_pnm_write(output, kind, rows, columns, maxcolor, image);
					printf("Load:\t\t%f\nScatter:\t%f\nReduction:\t%f\nScale:\t\t%f\nGather:\t\t%f\n",
						t_load[measurepos], t_scatter[measurepos], t_reduction[measurepos], t_scale[measurepos], t_gather[measurepos]);
				}
			
				/* MPI beenden */
				MPI_Finalize();
			break;}
		
			default:
			break;
		}
	}

	/*
	 * Calc average of the t_measures with MEASURES as count... 
	 */
	 
	
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
	int mins[omp_get_max_threads()];
	int maxs[omp_get_max_threads()];
	
	int i;
	for(i = 0; i < omp_get_max_threads(); i++) {
		mins[i] = maxcolor;
		maxs[i] = 0;
	}
	
	int x, y, threadnum;
	#pragma omp parallel for private(x, threadnum)
	for (y=0; y<rows; y++) {
		threadnum = omp_get_thread_num();
		
		for (x=0; x<columns; x++) {
			int color = image[y*columns+x];
			mins[threadnum] = MIN(mins[threadnum], color);
			mins[threadnum] = MAX(maxs[threadnum], color);
		}
	}
	
	for(i = 0; i < omp_get_max_threads(); i++) {
		mins[0] = MIN(mins[i], mins[0]);
		maxs[0] = MAX(maxs[i], maxs[0]);
	}
	
	*min = mins[0];
	*max = maxs[0];
}


void help() {
	printf("Usage:\n");
	printf("-c	Convert Type: 0 Sequential, 1 Openmp, 2 MPI \n");
	printf("-a	Integer for nmin\n");
	printf("-b	Integer for nmax\n");
	printf("-i	Path to input file\n");
	printf("-o	Path to output file\n");
}
