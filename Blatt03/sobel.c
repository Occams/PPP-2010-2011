#include <stdlib.h>
#include <sobel.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <ppp_pnm.h>

#define SOBEL_PIXEL(i,x,y,c,r) \
	((x < 0 || y < 0 || x >= c || y >= r) ? 0 : i[(y)*c+x])
	
#define SOBELX(i,x,y,c,r) \
	(SOBEL_PIXEL(i,x-1,y-1,c,r)+2*SOBEL_PIXEL(i,x,y-1,c,r)+SOBEL_PIXEL(i,x+1,y-1,c,r) \
	-SOBEL_PIXEL(i,x-1,y+1,c,r)-2*SOBEL_PIXEL(i,x,y+1,c,r)-SOBEL_PIXEL(i,x+1,y+1,c,r))

#define SOBELY(i,x,y,c,r) \
	(SOBEL_PIXEL(i,x-1,y-1,c,r)+2*SOBEL_PIXEL(i,x-1,y,c,r)+SOBEL_PIXEL(i,x-1,y+1,c,r) \
	-SOBEL_PIXEL(i,x+1,y-1,c,r)-2*SOBEL_PIXEL(i,x+1,y,c,r)-SOBEL_PIXEL(i,x+1,y+1,c,r))

static int sobel_mpi_self, sobel_mpi_processors;


void sobel_seq(int *image, int *dest, int rows, int columns, int c) {
	int x,y,sx,sy;

	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			sx = SOBELX(image,x,y,columns,rows);
			sy = SOBELY(image,x,y,columns,rows);
			dest[y*columns+x] = c*sqrt(sx*sx+sy*sy);
		}
	}
}

void sobel_mpi_init(int mpi_self, int mpi_processors) {
	sobel_mpi_self = mpi_self;
	sobel_mpi_processors = mpi_processors;
}

int *sobel_mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length) {
	int lines = rows/sobel_mpi_processors;
	int off = lines*sobel_mpi_self;
	int len = lines;
	
	if(sobel_mpi_self == 0) {
		len++;	// Line Below
	} else {
		off--;	// Line above
		if(sobel_mpi_self == sobel_mpi_processors - 1) {
			len++; // For line above
			len += rows%sobel_mpi_processors;
		} else {
			len += 2; // Line below + above
		}
	}
	
	*offset = off;
	*length = len;
	return (int*)malloc(lines*columns*sizeof(int));
}

void sobel_parallel(int *image, int *dest, int rows, int columns, int c) {
	int x,y,sx,sy;

	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			sx = SOBELX(image,x,y,columns,rows);
			sy = SOBELY(image,x,y,columns,rows);
			dest[y*columns+x] = c*sqrt(sx*sx+sy*sy);
		}
	}
}

void sobel_mpi_gather(void) {
}
