#include <vcd.h>

static double delta_edge(double *i, int x, int y, int rows, int cols);
static double delta(double *i, int x, int y, int rows, int cols);
static double s(double *img, int x, int y, int rows, int cols);
static void intToDoubleArray(int *src, double *dest, int length);
static void doubleToIntArray(double *src, int *dest, int length);

int *vcd_sequential(int *image, int rows, int columns, int maxcolor) {
	int i,x,y, rows_l = rows, columns_l = columns, idx, length = rows_l*columns_l;
	double img1[length], img2[length];
	double *img1_p = img1;
	double *img2_p = img2;
	double	d, *tmp;
	bool stop = false, edge;
	
	if (img1 == NULL || img2 == NULL) {
		printf("Out of memory.\n");
		exit(1);
	}
	
	intToDoubleArray(image, img1_p, length);
	
	for (i = 0; i < N && !stop; i++) {
		stop = true;
		
		for (x = 0; x < rows_l; x++) {
			for (y = 0; y < columns_l; y++) {
				idx = x*columns_l+y;
				edge = x == 0 || y == 0 || x+1 == rows_l || y+1 == columns_l;
				d = edge ? delta_edge(img1_p, x, y, rows_l, columns_l) : delta(img1_p, x, y, rows_l, columns_l);
				stop &= ABS(d) <= EPSILON || edge;	
				img2_p[idx] = img1_p[idx] + KAPPA * DELTA_T * d;
			}
		}
		
		tmp = img1_p;
		img1_p = img2_p;
		img2_p = tmp;
	}
	
	printf("VCD Iterations: %i", i);
	
	renormalize(img1, length, maxcolor);
	doubleToIntArray(img1, image, length);
	free(img1);
	free(img2);
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
	double center = i[x*cols + y];
	
	return PHI(i[(x+1)*cols + y] - center) 
		- PHI(center - i[(x-1)*cols + y])
		+ PHI(i[x*cols + y + 1] - center)
		- PHI(center - i[x*cols + y - 1])
		+ XI(i[(x+1)*cols + y + 1] - center)
		- XI(center - i[(x-1)*cols + y - 1])
		+ XI(i[(x-1)*cols + y + 1] - center)
		- XI(center - i[(x+1)*cols + y - 1]);

}

static double delta_edge(double *i, int x, int y, int rows, int cols) {
	double center = i[x*cols + y];
	
	return PHI(s(i,x+1,y,rows,cols) - center) 
		- PHI(center - s(i,x-1,y,rows,cols))
		+ PHI(s(i,x,y+1,rows,cols) - center)
		- PHI(center - s(i,x,y-1,rows,cols))
		+ XI(s(i,x+1,y+1,rows,cols) - center)
		- XI(center - s(i,x-1,y-1,rows,cols))
		+ XI(s(i,x-1,y+1,rows,cols) - center)
		- XI(center - s(i,x+1,y-1,rows,cols));
}

static double s(double *img, int x, int y, int rows, int cols) {
	return x < 0 || x >= rows || y < 0 || y >= cols ? 0.0 : img[x*cols+y];
}

