AR=ar
CC=gcc
CFLAGS=-g -std=gnu99 -Wall -O3 -fomit-frame-pointer
SDL_CFLAGS=`pkg-config --cflags sdl`
SDL_LDFLAGS=`pkg-config --libs sdl`

all: image_encoder viewer dct_flops

image_encoder: image_encoder.c libppp_pnm.a libppp_image.a
	$(CC) $(CFLAGS) -o $@ $< -L. -lppp_pnm -lppp_image -lm -lOpenCL

viewer: viewer.c libppp_pnm.a libppp_image.a
	$(CC) $(CFLAGS) -o $@ $< -L. -lppp_pnm -lppp_image $(SDL_CFLAGS) $(SDL_LDFLAGS)

dct_flops: dct_flops.c libppp_image.a
	$(CC) $(CFLAGS) -o $@ $< -L. -lppp_image -lm

libppp_image.a: ppp_image.o frame_encoding.o compression_stats.o ocl_init.o
	$(AR) r $@ $^

compression_stats.o: compression_stats.c compression_stats.h
	$(CC) $(CFLAGS) -c -o $@ $<

ppp_image.o: ppp_image.c ppp_image.h
	$(CC) $(CFLAGS) -c -o $@ $<

frame_encoding.o: frame_encoding.c
	$(CC) $(CFLAGS) -c -o $@ $<

frame_encoding.s: frame_encoding.c
	$(CC) $(CFLAGS) -S $<

ocl_init.o: ocl_init.c
	$(CC) $(CFLAGS) -c -o $@ $<

libppp_pnm.a: ppp_pnm.o
	$(AR) r $@ $^

ppp_pnm.o: ppp_pnm.c ppp_pnm.h
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f *.o *.s *.a image_encoder viewer dct_flops
