#include <sobel.h>
#include <math.h>

#define SOBELX(i,x,y) ((i)[y-1][x-1]+2*(i)[y-1][x]+(i)[y-1][x+1] - (i)[y+1][x-1] - 2*(i)[y+1][x] - (i)[y+1][x+1])
#define SOBELY(i,x,y) ((i)[y-1][x-1]+2*(i)[y][x-1]+(i)[y+1][x-1] - (i)[y-1][x+1] - 2*(i)[y][x+1] - (i)[y+1][x+1])

static int get_pixel(int x, int y);
static int tmp_rows, tmp_columns, *tmp_image;

void sobel_seq(int *image, int *dest, int rows, int columns, int c) {
	int x,y,sx,sy;
	tmp_rows = rows;
	tmp_columns = columns;
	tmp_image = image;

	for(y = 0; y < rows; y++) {
		for(x = 0; x < columns; x++) {
			sx = get_pixel(x-1,y-1)+2*get_pixel(x,y-1)+get_pixel(x+1,y-1)-get_pixel(x-1,y+1)-2*get_pixel(x,y+1)-get_pixel(x+1,y+1);
			sy = get_pixel(x-1,y-1)+2*get_pixel(x-1,y)+get_pixel(x-1,y+1)-get_pixel(x+1,y-1)-2*get_pixel(x+1,y)-get_pixel(x+1,y+1);
			dest[y*columns+x] = c+sqrt(sx*sx+sy*sy);
		}
	}
}

static int get_pixel(int x, int y) {
	if( x < 0 || y < 0 || y >= tmp_rows || x >= tmp_columns ) return 0;
	return tmp_image[y*tmp_columns+x];
}
