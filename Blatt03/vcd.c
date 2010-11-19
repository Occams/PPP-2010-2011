#include <vcd.h>
#include <math.h>
#include <stdbool.h>
#include "common.h"

static double delta(double *i, int x, int y, int rows, int cols);
static void renormalize(double *i, int length, int maxcolor);
static double s(double *img, int x, int y, int rows, int cols);
static void intToDoubleArray(int *src, double *dest, int length);
static void doubleToIntArray(double *src, int *dest, int length);

int *vcd_sequential(int *image, int rows, int columns, int maxcolor) {
	int i,x,y, rows_l = rows, columns_l = columns, idx, length = rows_l * columns_l;
	double img[length];
	bool stop = false;
	double d;
	
	intToDoubleArray(image, img, length);
	
	for (i = 0; i < N && !stop; i++) {
		stop =true;
		
		for (x = 0; x < rows_l; x++) {
			
			for (y = 0; y < columns_l; y++) {
				idx = y*columns_l+x;
				d = delta(img, x, y, rows_l, columns_l);
				
				/* Stop condition */
				stop &= d <= EPSILON || x == 0 || y == 0;	
				img[idx] = img[idx] + KAPPA * DELTA_T * d;
			}
		}
	}
	
	renormalize(img, length, maxcolor);
	doubleToIntArray(img, image, length);
	return image;
}

static void intToDoubleArray(int *src, double *dest, int length) {
	int x;
	
	for (x = 0; x < length; x++) {
		dest[x] = (int) src[x]; 
	}
}

static void doubleToIntArray(double *src, int *dest, int length) {
	int x;
	
	for (x = 0; x < length; x++) {
		dest[x] = (double) src[x]; 
	}
}

static double delta(double *i, int x, int y, int rows, int cols) {
	double center = s(i,x,y,rows,cols);
	
	return PHI(s(i,x+1,y,rows,cols) - center) 
		- PHI(center - s(i,x-1,y,rows,cols))
		+ PHI(s(i,x,y+1,rows,cols) - center)
		- PHI(center - s(i,x,y-1,rows,cols))
		+ XI(s(i,x+1,y+1,rows,cols) - center)
		- XI(center - s(i,x-1,y-1,rows,cols))
		+ XI(s(i,x-1,y+1,rows,cols) - center)
		- XI(center - s(i,x+1,y-1,rows,cols));
}

static void renormalize(double *i, int length, int maxcolor) {
	int x,length_l = length;
	double *img = i;
	
	for (x = 0; x < length_l; x++) {
			img[length] = MIN(img[length], maxcolor);
			img[length] = MAX(img[length], 0);
		}
}

static double s(double *img, int x, int y, int rows, int cols) {
	return x < 0 || x >= rows || y < 0 || y >= cols ? 0.0 : img[y*cols + x];
}

