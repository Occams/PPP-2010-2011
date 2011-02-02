#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <alloca.h>

#include "ppp_pnm.h"

#define MAXBUF 1024
#define IOBUF 16384

/*
 * Get an integer value (represented as decimal digits)
 * from a file; skip any comment lines (starting with '#')
 * and any non-digit characters in front of the integer value.
 * Exactly one character following the integer is read and
 * discarded, if that character is not '#'.
 */
static int getInt(FILE *fd)
{
    int i=0;
    char buffer[MAXBUF];
    do {
	*buffer = fgetc(fd);
	if (*buffer == '#') {
	    while (*buffer != EOF && *buffer != '\n')
		*buffer = fgetc(fd);
	}
    } while (*buffer != EOF && (*buffer < '0' || *buffer > '9'));

    while (i < MAXBUF-1 && buffer[i] >= '0' && buffer[i] <= '9')
	buffer[++i]=fgetc(fd);

    if (buffer[i] == '#')
	ungetc(buffer[i], fd);
    buffer[i]='\0';
    return atoi(buffer);
}

/*
 * Load the parameters from a PNM image file.
 */
static int ppp_pnm_params(FILE *fd, enum pnm_kind *kind, int *rows, int *cols,
			  int *maxval)
{
    if (fgetc(fd) != 'P') {
	return -1;
    }

    switch (fgetc(fd)) {
    case '4': *kind = PNM_KIND_PBM; break;
    case '5': *kind = PNM_KIND_PGM; break;
    case '6': *kind = PNM_KIND_PPM; break;
    default: return -1;
    }
    
    *cols = getInt(fd);
    *rows = getInt(fd);
    *maxval = (*kind == PNM_KIND_PBM) ? 1 : getInt(fd);
    
    return 0;
}


/*
 * Read image data from a PNM image in raw format.
 */
static int ppp_pnm_read_data(FILE *fd, enum pnm_kind kind,
			     int rows, int columns,
			     int maxval, uint8_t *image,
                             off_t offset, size_t length)
{
    int i, q, column;

    switch (kind) {
    case PNM_KIND_PBM:
	fseek(fd, offset/8, SEEK_CUR);
	column = offset % columns;
	q = 7 - offset%8;
	while (length > 0) {
	    int c = fgetc(fd);
	    for (i=q; i>=0; i--) {
		if (i < length && column < columns) {
		    image[offset++] = c & 1;
		    length--;
		    column++;
		}
		c >>= 1;
	    }
	    if (column == columns)
		column = 0;
	    q = 7;
	}
	break;
    case PNM_KIND_PGM: {
	fseek(fd, offset, SEEK_CUR);
        if (fread(image, sizeof(*image), length, fd) != length)
            return -1;
	break;
    }
    case PNM_KIND_PPM:
	fseek(fd, 3*offset, SEEK_CUR);
	length *= 3;
        if (fread(image, sizeof(*image), length, fd) != length)
            return -1;
	break;
    default: return -1;
    }
    return 0;
}

/*
 * 'part' function for ppp_pnm_read_part which reads
 * the whole image into a newly allocated array matching
 * exactly the image size (rows * columns).
 */
static uint8_t* read_all(enum pnm_kind kind, int rows, int columns,
                         int *offset, int *length)
{
    int valuesPerPixel;
    *offset = 0;
    *length = rows*columns;
    valuesPerPixel = kind == PNM_KIND_PPM ? 3 : 1;
    return (uint8_t *)malloc(sizeof(uint8_t) * *length * valuesPerPixel);
}

/*
 * Read a PNM image (PBM, PGM, PPM) from a file. Return
 * the image parameters in the kind, rows, cols, maxval arguments.
 * The image data is stored in row major order in the returned array.
 * PBM and PGM images use one entry per pixel, PPM use three
 * adjacent entries for the red, green, and blue channel.
 */
extern uint8_t *ppp_pnm_read(const char *name, enum pnm_kind *kind,
                             int *rows, int *columns, int *maxval)
{
    return ppp_pnm_read_part(name, kind, rows, columns, maxval, read_all);
}

extern uint8_t *ppp_pnm_read_part(const char *name, enum pnm_kind *kind,
                                  int *rows, int *columns, int *maxval,
                                  uint8_t*(*part)(enum pnm_kind, int, int,
                                                  int *, int *))
{
    FILE *fd;
    uint8_t *image;
    int offset, length;

    fd = fopen(name, "r");
    if (fd != NULL) {
	if (ppp_pnm_params(fd, kind, rows, columns, maxval) == 0) {
	    if (*maxval < 256) {
		image = part(*kind, *rows, *columns, &offset, &length);
		if (image != NULL) {
		    if (ppp_pnm_read_data(fd, *kind, *rows, *columns,
					  *maxval, image, offset, length) == 0) {
			fclose(fd);
			return image;
		    }
		    free(image);
		}
	    }
	}
	fclose(fd);
    }
    return NULL;
}

/*
 * Write a PNM image to a file.
 * The image is represented as described for ppp_pnm_read.
 */
extern int ppp_pnm_write(const char *name, enum pnm_kind kind, int rows, int cols, int maxval, uint8_t *image)
{
    FILE *fd;
    const char *creator = "# CREATOR: ppp_writepnm 2006-04-24";
    int i, j, c, len;

    fd = fopen(name,"w");
    if (fd == NULL)
	return errno;
    
    switch (kind) {
    case PNM_KIND_PBM:
	fprintf(fd,"P4\n%s\n%d %d\n", creator, cols, rows);
	for (j=0; j<rows; j++) {
	    i = 0;
	    for (i=0; i<cols; i+=8) {
		int b;
		c = 0;
		for (b=0; b<8; b++) {
		    if (i+b < cols)
			c = (c << 1) | (image[j*cols+i+b] != 0);
		}
		fputc(c, fd);
	    }
	}
	break;
    case PNM_KIND_PGM:
	fprintf(fd,"P5\n%s\n%d %d\n%d\n", creator, cols, rows, maxval);
	len = rows*cols;
        fwrite(image, sizeof(*image), len, fd);
	break;
    case PNM_KIND_PPM:
	fprintf(fd,"P6\n%s\n%d %d\n%d\n", creator, cols, rows, maxval);
	len = 3*rows*cols;
        fwrite(image, sizeof(*image), len, fd);
	break;
    default: fclose(fd); return -1;
    }
    
    fclose(fd);
    return 0;
}
