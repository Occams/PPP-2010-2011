#include <vcd.h>
#include <math.h>


int *vcd_sequential(int *image, int rows, int columns) {

	return image;
}

static double delta(double *i, int x, int y, int rows, int cols) {
	double center = s(i,x,y,rows,cols);
	
	return PHI(s(i,x+1,y,rows,cols)-s(i,x,y,rows,cols))
		- PHI(s(i,x,y,rows,cols));
}

static double s(double *img, int x, int y, int rows, int cols) {
	return x < 0 || x >= rows || y < 0 || y >= cols ? 0.0 : img[y*cols + x];
}

