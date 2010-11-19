#ifndef SOBEL_H_INCLUDED
#define SOBEL_H_INCLUDED

#define SOBEL_X(x,y,p) (p[y-1][x−1] + 2*p[y-1][x] + p[y-1][x+1] − p[y+1][x−1] − 2*p[y+1][x] − p[y+1][x+1])
#define SOBEL_Y(x,y,p) (p[y-1][x−1] + 2*p[y][x-1] + p[y+1][x-1] − p[y-1][x+1] − 2*p[y][x+1] − p[y+1][x+1])
#define SOBEL_SQRT(c,sx,sy) (c*sqrt(sx*sx+sy*sy))

void sobel_seq(int *image, int *dest, int rows, int columns);

#endif
