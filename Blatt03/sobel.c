#include <stdlib.h>
#include <stdio.h>
#include <sobel.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <ppp_pnm.h>
#include <pgm_distribute.h>

#define SOBEL_PIXEL(i,x,y,c,r) \
	((x < 0 || y < 0 || x >= c || y >= r) ? 0 : i[(y)*c+x])
	
#define SOBELX(i,x,y,c,r) \
	(SOBEL_PIXEL(i,x-1,y-1,c,r)+2*SOBEL_PIXEL(i,x,y-1,c,r)+SOBEL_PIXEL(i,x+1,y-1,c,r) \
	-SOBEL_PIXEL(i,x-1,y+1,c,r)-2*SOBEL_PIXEL(i,x,y+1,c,r)-SOBEL_PIXEL(i,x+1,y+1,c,r))

#define SOBELY(i,x,y,c,r) \
	(SOBEL_PIXEL(i,x-1,y-1,c,r)+2*SOBEL_PIXEL(i,x-1,y,c,r)+SOBEL_PIXEL(i,x-1,y+1,c,r) \
	-SOBEL_PIXEL(i,x+1,y-1,c,r)-2*SOBEL_PIXEL(i,x+1,y,c,r)-SOBEL_PIXEL(i,x+1,y+1,c,r))

static int sobel_mpi_self, sobel_mpi_processors;

void sobel_seq(int *image, int rows, int columns, int c, int maxcolor) {
	int x,y,sx,sy;
	int dest[rows*columns];

	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			sx = SOBELX(image,x,y,columns,rows);
			sy = SOBELY(image,x,y,columns,rows);
			dest[y*columns+x] = c*sqrt(sx*sx+sy*sy);
		}
	}
	
	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			image[y*columns+x] = MIN(dest[y*columns+x], maxcolor);
			image[y*columns+x] = MAX(dest[y*columns+x], 0);
		}
	}
}

void sobel_mpi_init(int mpi_self, int mpi_processors) {
	sobel_mpi_self = mpi_self;
	sobel_mpi_processors = mpi_processors;
}


int *sobel_mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length) {
	pgm_part info;
	pgm_partinfo(rows, sobel_mpi_self, &info);
	
	*offset = info.offset*columns;
	*length = info.rows*columns;
	
	return (int*)malloc(info.rows*columns*sizeof(int));
}

void sobel_parallel(int *image, int rows, int columns, int c, int maxcolor) {
	int x,y,sx,sy,idx;
	int dest[rows*columns];

	#pragma omp parallel for private(x)
	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			sx = SOBELX(image,x,y,columns,rows);
			sy = SOBELY(image,x,y,columns,rows);
			dest[y*columns+x] = c*sqrt(sx*sx+sy*sy);
		}
	}
	
	
	#pragma omp parallel for private(x,idx)
	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			idx = y*columns+x;
			image[idx] = MIN(dest[idx], maxcolor);
			image[idx] = MAX(dest[idx], 0);
		}
	}
}
