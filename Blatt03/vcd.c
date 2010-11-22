#include <vcd.h>

static int vcd_mpi_self = 0, vcd_mpi_processors = 1;
int vcd_stop = false;

static double delta_edge(double *i, int x, int y, int rows, int cols);
static double delta(double *i, int x, int y, int rows, int cols);
static double s(double *img, int x, int y, int rows, int cols);
static void intToDoubleArray(int *src, double *dest, int length);
static void doubleToIntArray(double *src, int *dest, int length);
static void intToDoubleArray_parallel(int *src, double *dest, int length);
static void doubleToIntArray_parallel(double *src, int *dest, int length);

void vcd_parallel(int *image, int rows, int columns, int maxcolor) {
	MPI_Status status;
	MPI_Request request;
	int i,x,y, rows_l = vcd_mpi_self < vcd_mpi_processors-1 ? rows-1 : rows, idx, length = rows*columns;
	double img1[length], img2[length], *tmp, *img1_p = img1, *img2_p = img2, d;
	bool edge;
	
	intToDoubleArray_parallel(image, img1_p, length);
	
	for (i = 0; i < N && !vcd_stop; i++) {
		vcd_stop = true;
		
		#pragma omp parallel for private (y,idx,edge,d) reduction (&& : vcd_stop)
		for (x = vcd_mpi_self > 0 ? 1 : 0; x < rows_l; x++) {
			for (y = 0; y < columns; y++) {
				idx = x*columns+y;
				edge = x == 0 || y == 0 || x+1 == rows || y+1 == columns;
				d = edge ? delta_edge(img1_p, x, y, rows, columns) : delta(img1_p, x, y, rows, columns);
				vcd_stop = vcd_stop && (edge || ABS(d) <= EPSILON);
				img2_p[idx] = img1_p[idx] + KAPPA * DELTA_T * d;
			}
		}
	}
	
	/* Switch array pointers */
	tmp = img1_p;
	img1_p = img2_p;
	img2_p = tmp;
	
	/* Update stop condition */
	MPI_Allreduce(&vcd_stop,&vcd_stop,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
	
	/* Share overlapping at the top */
	if(vcd_mpi_self > 0) {
		MPI_Isend(img1_p+columns, columns, MPI_DOUBLE, vcd_mpi_self - 1,
		0, MPI_COMM_WORLD, &request);
	}
	
	if (vcd_mpi_self < vcd_mpi_processors-1) {
		MPI_Recv(img1_p+(rows-1)*columns, columns, MPI_DOUBLE, vcd_mpi_self + 1,
		0, MPI_COMM_WORLD, &status);
	}
	
	/* Share overlapping at the bottom */
	if(vcd_mpi_self < vcd_mpi_processors-1) {
		MPI_Isend(img1_p+(rows-2)*columns, columns, MPI_DOUBLE, vcd_mpi_self + 1,
		1, MPI_COMM_WORLD, &request);
	}
	
	if(vcd_mpi_self > 0) {
		MPI_Recv(img1_p, columns, MPI_DOUBLE, vcd_mpi_self - 1,
		1, MPI_COMM_WORLD, &status);
	}
	
	printf("VCD Iterations: %i\n", i);
	doubleToIntArray_parallel(img1_p, image, length);
}

void vcd_sequential(int *image, int rows, int columns, int maxcolor) {
	int i, x, y, idx, length = rows*columns;
	double img1[length], img2[length], *tmp, *img1_p = img1, *img2_p = img2, d;
	bool edge;
	
	intToDoubleArray(image, img1_p, length);
	
	for (i = 0; i < N && !vcd_stop; i++) {
		vcd_stop = true;
		
		for (x = 0; x < rows; x++) {
			for (y = 0; y < columns; y++) {
				idx = x*columns+y;
				edge = x == 0 || y == 0 || x+1 == rows || y+1 == columns;
				d = edge ? delta_edge(img1_p, x, y, rows, columns) : delta(img1_p, x, y, rows, columns);
				vcd_stop = vcd_stop && (edge || ABS(d) <= EPSILON);	
				img2_p[idx] = img1_p[idx] + KAPPA * DELTA_T * d;
			}
		}
		
		tmp = img1_p;
		img1_p = img2_p;
		img2_p = tmp;
	}
	
	printf("VCD Iterations: %i\n", i);
	
	doubleToIntArray(img1_p, image, length);
	pgm_renormalize(image, length, maxcolor);
}

static void intToDoubleArray(int *src, double *dest, int length) {
	int x;
	//double start = seconds();
	for (x = 0; x < length; x++) {
		dest[x] = (int) src[x];
	}
	//printf("Intodouble seq: %f\n",seconds() - start);
}

static void intToDoubleArray_parallel(int *src, double *dest, int length) {
	int x;
	//double start = seconds();
	#pragma omp parallel for
	for (x = 0; x < length; x++) {
		dest[x] = (int) src[x];
	}
	//printf("Intodouble parallel: %f\n",seconds() - start);
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

