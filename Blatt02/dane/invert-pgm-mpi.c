#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "ppp_pnm.h"

int self, np;
int myLength;

/*
 * Load a part of an image on the current processor
 */
int *partfn(enum pnm_kind kind,
            int rows, int columns,
            int *offset, int *length)
{
    if (kind != PNM_KIND_PGM)
	return NULL;

    /*
     * The number of pixels need not be a multiple of
     * np. Therefore, the first  (rows*columns)%np  processes get
     *    ceil((rows*columns)/np)
     * pixels, and the remaining processes get
     *    floor((rows*columns)/np)
     * pixels.
     */
    if (self < (rows*columns)%np) {
	*length = (rows*columns)/np + 1;
	*offset = *length * self;
    } else {
	*length = (rows*columns)/np;
	*offset = *length * self  +  (rows*columns)%np;
    }

    myLength = *length;

    printf("self=%d, offset=%d, length=%d\n", self, *offset, *length);

    /*
     * Allocate space for the image part.
     * On processor 0 we allocate space for the whole
     * result image.
     */
    return (int*)malloc((self == 0 ? rows*columns : myLength) * sizeof(int));
}

/*
 * Load a PGM (Portable Graymap) image and invert
 * the gray values of every pixel.
 * The program is called with 2 arguments:
 *      Input-image  Output-image
 */
int main(int argc, char *argv[]) {
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    int *myPart;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (argc != 3) {
	printf("USAGE: %s IN OUT\n", argv[0]);
	return 1;
    }

    /*
     * Load a part of an image. Parameters are as for
     * ppp_pnm_read; the last parameter is a callback function
     * which tells ppp_pnm which part of the image to load. 
     */
    myPart = ppp_pnm_read_part(argv[1], &kind, &rows, &columns, &maxcolor,
			       partfn);
    if (myPart != NULL) {
	int i;
	for (i=0; i<myLength; i++)
	    myPart[i] = maxcolor - myPart[i];

	/*
	 * Collect the image parts on processor 0 and save
	 * them to the output file. MPI_Gather would be more
	 * efficient. Note that when the image parts have
         * different sizes (because np does not divide rows*columns)
         * then one should use MPI_Gatherv to collect the
         * parts. MPI_Gatherv allows parts with different sizes
         * whereas MPI_Gather assumes that all parts have the same
         * size.
	 */
	if (self == 0) {
	    MPI_Status status;
	    int offset, len;

	    offset = myLength;
	    for (i=1; i<np; i++) {
		len = (rows*columns)/np;
		if (i < (rows*columns)%np)
		    len++;
		MPI_Recv(myPart+offset, len, MPI_INT, i, MPI_ANY_TAG,
			 MPI_COMM_WORLD, &status);
		offset += len;
	    }

	    if (ppp_pnm_write(argv[2], kind, rows, columns, maxcolor, myPart) != 0)
		printf("write error\n");
	} else
	    MPI_Send(myPart, myLength, MPI_INT, 0, 0, MPI_COMM_WORLD);

	free(myPart);
    } else
	printf("could not load image\n");

    MPI_Finalize();

    return 0;
}
