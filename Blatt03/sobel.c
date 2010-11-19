#include <sobel.h>
#include <math.h>

#define SOBEL_PIXEL(i,x,y,c,r) \
	((x < 0 || y < 0 || x >= c || y >= r) ? 0 : i[(y)*c+x])
	
#define SOBELX(i,x,y,c,r) \
	(SOBEL_PIXEL(i,x-1,y-1,c,r)+2*SOBEL_PIXEL(i,x,y-1,c,r)+SOBEL_PIXEL(i,x+1,y-1,c,r) \
	-SOBEL_PIXEL(i,x-1,y+1,c,r)-2*SOBEL_PIXEL(i,x,y+1,c,r)-SOBEL_PIXEL(i,x+1,y+1,c,r))

#define SOBELY(i,x,y,c,r) \
	(SOBEL_PIXEL(i,x-1,y-1,c,r)+2*SOBEL_PIXEL(i,x-1,y,c,r)+SOBEL_PIXEL(i,x-1,y+1,c,r) \
	-SOBEL_PIXEL(i,x+1,y-1,c,r)-2*SOBEL_PIXEL(i,x+1,y,c,r)-SOBEL_PIXEL(i,x+1,y+1,c,r))



void sobel_seq(int *image, int *dest, int rows, int columns, int c) {
	int x,y,sx,sy;

	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			sx = SOBELX(image,x,y,columns,rows);
			sy = SOBELY(image,x,y,columns,rows);
			dest[y*columns+x] = c*sqrt(sx*sx+sy*sy);
		}
	}
}
