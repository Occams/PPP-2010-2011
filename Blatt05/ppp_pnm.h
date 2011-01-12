#ifndef PPP_PNM_H
#define PPP_PNM_H

#include <stdint.h>

enum pnm_kind {
    PNM_KIND_PBM,   /* Portable Bitmap (black/white)    */
    PNM_KIND_PGM,   /* Portable Graymap                 */
    PNM_KIND_PPM    /* Portable Pixmap (red/green/blue) */
};

/*
 * Read a PNM image (PBM, PGM, PPM) from a file. Return
 * the image parameters in the kind, rows, cols, maxval arguments.
 * The image data is stored in row major order in the returned array.
 * PBM and PGM images use one entry per pixel, PPM use three
 * adjacent entries for the red, green, and blue channel.
 */
extern uint8_t *ppp_pnm_read(const char *name, enum pnm_kind *kind,
                             int *rows, int *columns, int *maxval);

/*
 * Read a part of a PNM image.
 * name, kind, rows, maxval are as for ppp_pnm_read.
 * partfn is a pointer to a function which is called with
 * kind, rows and columns of the image; it is expected to set
 * values for *offset and *length which describe the part
 * of the image to be loaded (measured in pixels). partfn must
 * return a pointer to buffer where the image part is to be stored.
 * calloc or malloc must be used to allocate the buffer.
 * ppp_pnm_read_part returns the pointer returned by partfn.
 */
extern uint8_t *ppp_pnm_read_part(const char *name, enum pnm_kind *kind,
                                  int *rows, int *columns, int *maxval,
                                  uint8_t *(*partfn)(enum pnm_kind kind,
                                                     int rows, int columns,
                                                     int *offset, int *length) );

/*
 * Write an PNM image to a file.
 * The image is represented as described for ppp_pnm_read.
 */
extern int ppp_pnm_write(const char *name, enum pnm_kind kind,
			 int rows, int columns, int maxval,
                         uint8_t *image);


#endif
