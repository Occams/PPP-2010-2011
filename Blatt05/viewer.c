#include <getopt.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <SDL/SDL.h>
#include <SDL/SDL_thread.h>

#include "ppp_pnm.h"
#include "frame_encoding.h"

/*
 * Convert a pixel value 'val' (in range -128..127) to
 * a gray value for the screen (in range 0..255). If the value
 * is out of range, use 0 or 255, respectively.
 */
static uint8_t int16_to_pixel(int16_t val) {
    if (val < -128)
        return 0;
    else if (val > 127)
        return 255;
    return (uint8_t)(val + 128);
}

/*
 * Decode a macro block.
 * The compressed data is stored in 'data', the frame parameters
 * are given in 'info'. The decoded block is stored in 'image'
 * with upper left corner at (yy,xx).
 * Return the number of bytes consumed from 'data'.
 */
int decode_block(const uint8_t *data, const ppp_image_info *info,
                 uint8_t *image, int yy, int xx) {
    int16_t data16[64], decoded[64];
    const int columns = info->columns;
    int off, len;
    /* 'img' points to the upper left corner of the block to decode
     * in 'image'. */
    uint8_t * const img = image + yy*columns + xx;

    switch (info->format) {
    case PPP_IMGFMT_UNCOMPRESSED_BLOCKS:
        for (int y=0; y<8; y++)
            for (int x=0; x<8; x++)
                img[y*columns+x] = (uint8_t)(*data++ + 128);
        return 64;
        break;
    case PPP_IMGFMT_UNCOMPRESSED_DCT:
        for (int i=0; i<64; i++)
            data16[i] = ((int8_t *)data)[i];
        iqdct_block(data16, decoded);
        off = 0;
        for (int y=0; y<8; y++)
            for (int x=0; x<8; x++)
                img[y*columns+x] = int16_to_pixel(decoded[off++]);
        return 64;
        break;
    case PPP_IMGFMT_COMPRESSED_DCT:
        len = uncompress_block(data, data16);
        iqdct_block(data16, decoded);
        off = 0;

        for (int y=0; y<8; y++) 
            for (int x=0; x<8; x++)
                img[y*columns+x] = int16_to_pixel(decoded[off++]);

        return len;
        break;
    }
    return 0;
}

void decode_frame(const ppp_frame *frame, const ppp_image_info *info,
                  uint8_t *image) {
    const int rows = info->rows;
    const int columns = info->columns;

    int b = 0, off = 0;
    for (int yy=0; yy<rows; yy+=8) {
        for (int xx=0; xx<columns; xx+=8) {
            off += decode_block(frame->data+off, info,
                                image, yy, xx);
            b++;
        }
    }
}


typedef struct {
    SDL_Overlay *overlay;
    uint8_t *image;
    ppp_image_info img_info;
    float fps;
    ppp_frame *frame;
} display_info;

/*
 * Redraw the image on the screen.
 */
void redraw(display_info *di) {
    if (di->image != NULL) {
        SDL_Overlay *overlay = di->overlay;
        SDL_LockYUVOverlay(overlay);
        uint8_t *y = (uint8_t *)overlay->pixels[0];
        int linesizeY = overlay->pitches[0];
        int columns = di->img_info.columns;
        /* Copy the rows of 'image' into the SDL overlay. */
        for (int r=0; r<di->img_info.rows; r++)
            memcpy(y+linesizeY*r, di->image+columns*r, columns);
        SDL_UnlockYUVOverlay(overlay);
        SDL_Rect rect;
        rect.x = 0;
	rect.y = 0;
	rect.w = di->img_info.columns;
	rect.h = di->img_info.rows;
	SDL_DisplayYUVOverlay(overlay, &rect);
    }
}    


/*
 * Load file 'filename' and store parameters is 'di'.
 * Return 0 on success.
 */
int load(const char *filename, display_info *di) {
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image;

    image = ppp_pnm_read(filename, &kind, &rows, &columns, &maxcolor);  
    if (image != NULL) {
        if (kind != PNM_KIND_PGM)
            return 1;
        di->image = image;
        di->img_info.rows    = rows;
        di->img_info.columns = columns;
        return di->image != NULL ? 0 : 1;
    }
    
    ppp_frame *frame = ppp_image_read(filename, &(di->img_info));
    if (frame != NULL) {
        int pixels = di->img_info.rows*di->img_info.columns;
        di->image = (uint8_t *)malloc(pixels * sizeof(uint8_t));
        if (di->image != NULL)
            decode_frame(frame, &(di->img_info), di->image);
        free(frame);
        return di->image == NULL ? 1 : 0;
    }

    return 1;
}

void usage(const char *progname) {
    fprintf(stderr, "USAGE: %s [-s] FILE\n"
            "  FILE can be a (binary encoded) PGM image or a PPPI image.\n"
            "\n",
            progname);
}

int main( int   argc,
          char *argv[] )
{
    const char *filename;
    int option, single_step;
    display_info di;
    SDL_Surface *screen;
    SDL_Overlay *overlay;

    init_qdct();

    single_step = 0;
    while ((option = getopt(argc,argv,"s")) != -1) {
        switch(option) {
        case 's': single_step = 1; break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (optind < argc)
        filename = argv[optind];
    else {
        usage(argv[0]);
        return 1;
    }

    if (load(filename, &di) != 0) {
        fprintf(stderr, "Could not load file '%s'.\n", filename);
        exit(1);
    }
    
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER)) {
        fprintf(stderr, "Could not initialize SDL - %s\n", SDL_GetError());
        exit(1);
    }

    screen = SDL_SetVideoMode(di.img_info.columns, di.img_info.rows, 0, 0);
    if (screen == NULL) {
        fprintf(stderr, "SDL: could not set video mode - exiting\n");
        exit(1);
    }

    overlay = SDL_CreateYUVOverlay(di.img_info.columns, di.img_info.rows,
                                   SDL_YV12_OVERLAY, screen);
    if (overlay == NULL) {
        fprintf(stderr, "SDL: could not create overlay\n");
        exit(1);
    }
    di.overlay = overlay;

    SDL_WM_SetCaption(filename, "PPP Viewer");

    /* Initialize the U and V components to 128 so the
     * Y component determines a gray value (when we set it later).
     */
    uint8_t *u = (uint8_t *)overlay->pixels[1];
    uint8_t *v = (uint8_t *)overlay->pixels[2];
    int linesizeU = overlay->pitches[1];
    int linesizeV = overlay->pitches[2];
    int columns = di.img_info.columns;
    for (int r=0; r<(di.img_info.rows+1)/2; r++) {
        memset(u+linesizeU*r, 128, (columns+1)/2);
        memset(v+linesizeV*r, 128, (columns+1)/2);
    }

    redraw(&di);

    SDL_Event event;
    int cont = 1;
    while (cont && SDL_WaitEvent(&event)) {  
        switch (event.type)
        {
        case SDL_VIDEOEXPOSE: redraw(&di); break;
        case SDL_KEYDOWN:
            switch (event.key.keysym.sym) {
            case SDLK_ESCAPE:
            case SDLK_q:
                cont = 0;
                break;
            default: break;
            }
            break;
        case SDL_QUIT: cont = 0; break;
        }
    }

    SDL_FreeYUVOverlay(overlay);
    SDL_FreeSurface(screen);
    
    return 0;
}
