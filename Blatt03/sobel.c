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
	int x,y,sx,sy,pixel;
	int dest[rows*columns];

	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			sx = SOBELX(image,x,y,columns,rows);
			sy = SOBELY(image,x,y,columns,rows);
			pixel = c*sqrt(sx*sx+sy*sy);
			dest[y*columns+x] = pixel < 0 ? 0 : MIN(pixel, maxcolor);
		}
	}
	
	for(y = 0; y < rows*columns; y++) {
		image[y] = dest[y];
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
	int x,y,sx,sy,pixel;
	int dest[rows*columns];
	
	#pragma omp parallel for private(x,pixel)
	for(y = sobel_mpi_self > 0 ? 1 : 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			sx = SOBELX(image,x,y,columns,rows);
			sy = SOBELY(image,x,y,columns,rows);
			pixel = c*sqrt(sx*sx+sy*sy);
			dest[y*columns+x] = pixel < 0 ? 0 : MIN(pixel, maxcolor);
		}
	}
	
	#pragma omp parallel for
	for(y = 0; y < rows*columns; y++) {
		image[y] = dest[y];
	}
	
}
