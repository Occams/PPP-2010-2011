#include <sobel.h>
#include <math.h>

static void get_octet(int* image, int *dest, int x, int y, int rows, int cols);

void sobel_seq(int *image, int *dest, int rows, int columns, int c) {
	int x,y,sx,sy;
	int octet[9] = {0};

	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			//sx = SOBEL_X(x,y,image);
			//sy = SOBEL_Y(x,y,image);
			dest[y*columns+x] = SOBEL_SQRT(c,sx,sy);
		}
	}
}

static void get_octet(int* image, int *dest, int x, int y, int rows, int cols) {
}
