#include <vcd.h>

static int vcd_mpi_self = 0, vcd_mpi_processors = 1;

static double delta_edge(double *i, int x, int y, int rows, int cols);
static double delta(double *i, int x, int y, int rows, int cols);
static double s(double *img, int x, int y, int rows, int cols);
static void intToDoubleArray(int *src, double *dest, int length);
static void doubleToIntArray(double *src, int *dest, int length);
static void intToDoubleArray_parallel(int *src, double *dest, int length);
static void doubleToIntArray_parallel(double *src, int *dest, int length);

int *vcd_parallel(int *image, int rows, int columns, int maxcolor) {
	int i,x,y, rows_l = rows, columns_l = columns, idx, length = rows_l*columns_l;
	double *img1 = (double *) malloc(length*sizeof(double)), *img2 = (double *) malloc(length*sizeof(double)), *tmp, d;
	bool stop = false, edge;
	
	if (img1 == NULL || img2 == NULL) {
		printf("Out of memory.\n");
		exit(1);
	}
	
	
}

int *vcd_sequential(int *image, int rows, int columns, int maxcolor) {
	int i,x,y, rows_l = rows, columns_l = columns, idx, length = rows_l*columns_l;
	double *img1 = (double *) malloc(length*sizeof(double)), *img2 = (double *) malloc(length*sizeof(double)), *tmp, d;
	bool stop = false, edge;
	
	if (img1 == NULL || img2 == NULL) {
		printf("Out of memory.\n");
		exit(1);
	}
	
	intToDoubleArray(image, img1, length);
	
	for (i = 0; i < N && !stop; i++) {
		stop = true;
		
		for (x = 0; x < rows_l; x++) {
			for (y = 0; y < columns_l; y++) {
				idx = x*columns_l+y;
				edge = x == 0 || y == 0 || x+1 == rows_l || y+1 == columns_l;
				d = edge ? delta_edge(img1, x, y, rows_l, columns_l) : delta(img1, x, y, rows_l, columns_l);
				stop = stop && ( ABS(d) <= EPSILON || edge);	
				img2[idx] = img1[idx] + KAPPA * DELTA_T * d;
			}
		}
		
		tmp = img1;
		img1 = img2;
		img2 = tmp;
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

static void intToDoubleArray_parallel(int *src, double *dest, int length) {
	int x;
	
	#pragma omp parallel for
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

static void doubleToIntArray_parallel(double *src, int *dest, int length) {
	int x;
	
	#pragma omp parallel for
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

void vcd_mpi_init(int mpi_self, int mpi_processors) {
	vcd_mpi_self = mpi_self;
	vcd_mpi_processors = mpi_processors;
}


int *vcd_mpi_read_part(enum pnm_kind kind, int rows, int columns, int *offset, int *length) {
	pgm_part info;
	pgm_partinfo(rows, vcd_mpi_self, &info);
	
	*offset = info.offset*columns;
	*length = info.rows*columns;
	
	return (int*)malloc(info.rows*columns*sizeof(int));
}

