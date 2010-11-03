#include <stdio.h>
#include <stdlib.h>
#include "ppp_pnm.h"

/*
 * Declarations
 */


/*
 * Implementations
 */
int main(int argc, char *argv[]) {
	int amin, amax;
	amin = 255;
	amax = 0;
	
	/*
	 * Open image and read out the amin and amax values.
	 */
	enum pnm_kind kind;
	int rows, columns, maxcolor;
	int* image;
	image = ppp_pnm_read(argv[1], &kind, &rows, &columns, &maxcolor);
	
	if(image != NULL) {
		if(kind == PNM_KIND_PGM) {
			for (y=0; y<rows; y++) {
				for (x=0; x<columns; x++) {
					int color = image[y*columns+x];
					amin = amin<color?amin:color;
					amax = amax>color?amax:color;
				}
			}
		}
		
		printf("Minimum: %i\nMaximum: %i\n\n", amin, amax);
	    
	    free(image);
	}
	
	
}
