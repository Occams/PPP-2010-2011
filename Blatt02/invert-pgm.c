/*
 * Beispiel fuer die Benutzung der ppp_pnm Bibliothek.
 * Kompilieren auf hermes mit:
 *     gcc -Wall -o invert-pgm invert-pgm.c
 *          -I/home/hpcuser/ppp2010-data/packages/ppp_pnm-32
 *          -L/home/hpcuser/ppp2010-data/packages/ppp_pnm-32
 *          -lppp_pnm
 *
 * Auf ravel/hydra statt ppp_pnm-32 die 64-bit Variante ppp_pnm-64 benutzen.
 */
#include <stdio.h>
#include <stdlib.h>
#include "ppp_pnm.h"

/*
 * Load a PGM (Portable Graymap) image and invert
 * the gray values of every pixel.
 * The program is called with 2 arguments:
 *      Input-image  Output-image
 */
int main(int argc, char *argv[]) {
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    int *image;
    int x, y;

    if (argc != 3) {
	printf("USAGE: %s IN OUT\n", argv[0]);
	return 1;
    }

    /*
     * Load the image (name in argv[1]),
     * store the kind (PBM, PGM, PPM) in 'kind',
     * the number of rows and columns in 'rows' and 'columns',
     * the maximal gray value of the image format (NOT the
     * maximal gray value used in the image) in 'maxcolor' and return
     * the image row-wise with one int per pixel.
     */
    image = ppp_pnm_read(argv[1], &kind, &rows, &columns, &maxcolor);
    
    if (image != NULL) {
	if (kind == PNM_KIND_PGM) {
	    for (y=0; y<rows; y++) {
		for (x=0; x<columns; x++) {
		    image[y*columns+x] = maxcolor - image[y*columns+x];
		}
	    }
	    
	    /*
	     * Save the image, parameters are analogous to
	     * ppp_pnm_read (but kind, rows, columns, maxcolor are
	     * not passed as pointers for ppp_pnm_write). The
	     * last parameter is a pointer to the image to be saved.
	     */
	    if (ppp_pnm_write(argv[2], kind, rows, columns, maxcolor, image) != 0)
		printf("write error\n");
	} else
	    printf("not a PGM image\n");

	free(image);
    } else
	printf("could not load image\n");

    return 0;
}
