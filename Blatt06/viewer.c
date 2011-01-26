#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <SDL/SDL.h>
#include <SDL/SDL_thread.h>

#include "ppp_pnm.h"
#include "ppp_video.h"
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
 * 'motion' and 'old_image' are used for prediction/motion compensation.
 * Return the number of bytes consumed from 'data'.
 */
int decode_block(const uint8_t *data, int max_input_len,
                 ppp_motion motion, const ppp_image_info *info,
                 const uint8_t *old_image, uint8_t *image, int yy, int xx) {
    int16_t data16[64], decoded[64];
    const int columns = info->columns;
    const int rows = info->rows;
    int deltaX, deltaY;
    int off, len;
    bool dec_error;
    /* 'img' points to the upper left corner of the block to decode
     * in 'image'. */
    uint8_t * const img = image + yy*columns + xx;

    switch (info->format) {
    case PPP_IMGFMT_UNCOMPRESSED_BLOCKS:
        if (max_input_len < 64) {
            fprintf(stderr, "Warning: frame too short\n");
            return max_input_len;
        }
        for (int y=0; y<8; y++)
            for (int x=0; x<8; x++)
                img[y*columns+x] = (uint8_t)(*data++ + 128);
        return 64;
    case PPP_IMGFMT_UNCOMPRESSED_DCT:
        if (max_input_len < 64) {
            fprintf(stderr, "Warning: frame too short\n");
            return max_input_len;
        }
        for (int i=0; i<64; i++)
            data16[i] = ((int8_t *)data)[i];
        iqdct_block(data16, decoded);
        off = 0;
        for (int y=0; y<8; y++)
            for (int x=0; x<8; x++)
                img[y*columns+x] = int16_to_pixel(decoded[off++]);
        return 64;
    case PPP_IMGFMT_COMPRESSED_DCT:
        len = uncompress_data(data, max_input_len, 64, data16, &dec_error);
        if (dec_error)
            fprintf(stderr, "Warning: block uncompression error for block at "
                    "top=%d, left=%d\n", yy, xx);
        iqdct_block(data16, decoded);
        off = 0;

        if (is_intra(motion))
            deltaX = deltaY = 0;
        else
            decode_motion_code(motion, &deltaY, &deltaX);
        
        if (yy+deltaY < 0 || xx+deltaX < 0 || yy+7+deltaY >= rows || xx+7+deltaX >= columns) {
            fprintf(stderr, "erroneous motion vector: yy=%d, xx=%d, deltaY=%d, deltaX=%d\n", yy, xx, deltaY, deltaX);
            deltaX = deltaY = 0;
        }

        if (is_intra(motion)) {
            for (int y=0; y<8; y++) 
                for (int x=0; x<8; x++)
                    img[y*columns+x] = int16_to_pixel(decoded[off++]);

        } else {
            for (int y=0; y<8; y++) {
                for (int x=0; x<8; x++) {
                    int pixel = (yy+y+deltaY)*columns + xx+x+deltaX;
                    int16_t v = (int16_t)old_image[pixel] - 128;
                    img[y*columns+x] = int16_to_pixel(v + decoded[off++]);
                }
            }
        }
        return len;
        break;
    }
    return 0;
}

void decode_video_frame(const ppp_frame *frame, const ppp_motion *motions,
                        const ppp_image_info *info,
                        const uint8_t *old_image, uint8_t *image) {
    const int rows = info->rows;
    const int columns = info->columns;

    int b = 0, off = 0;
    for (int yy=0; yy<rows; yy+=8) {
        for (int xx=0; xx<columns; xx+=8) {
            const ppp_motion bm = 
                motions == NULL ? PPP_MOTION_INTRA : motions[b];
            off += decode_block(frame->data+off, frame->length-off,
                                bm, info, old_image, image, yy, xx);
            b++;
        }
    }
}


enum video_type {
    VT_NONE,    /* image */
    VT_VL,      /* ppp_video_load */
    VT_PPP      /* our own format */
};

typedef struct {
    SDL_Overlay *overlay;
    uint8_t *image, *image2;
    enum video_type vid_type;
    ppp_image_info img_info;
    float fps;
    ppp_frame *frame;

    union {
        FILE *file;
        video *video;
    };
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
        di->image2 = NULL;
        di->img_info.rows    = rows;
        di->img_info.columns = columns;
        di->vid_type = VT_NONE;
        return di->image != NULL ? 0 : 1;
    }
    
    ppp_frame *frame = ppp_image_read(filename, &(di->img_info));
    if (frame != NULL) {
        int pixels = di->img_info.rows*di->img_info.columns;
        di->image = (uint8_t *)malloc(pixels * sizeof(uint8_t));
        if (di->image != NULL)
            decode_video_frame(frame, NULL, &(di->img_info), NULL, di->image);
        di->image2 = NULL;
        free(frame);
        di->vid_type = VT_NONE;
        return di->image == NULL ? 1 : 0;
    }

    ppp_video_info vid_info;
    FILE *f = ppp_video_read(filename, &(di->img_info), &vid_info);
    if (f != NULL) {
        int pixels = di->img_info.rows*di->img_info.columns;
        di->image = (uint8_t *)malloc(2 * pixels * sizeof(uint8_t));
        if (di->image == NULL) {
            fclose(di->file);
            return 1;
        }
        memset(di->image, 0, 2*sizeof(uint8_t)*pixels);
        di->image2   = di->image + pixels;
        di->file     = f;
        di->vid_type = VT_PPP;
        di->fps      = vid_info.fps / 1000.0f;
        di->frame    = ppp_frame_alloc(max_encoded_length(pixels));
        return 0;
    }

    video *v = video_open(filename);
    if (v != NULL) {
        di->img_info.rows   = video_get_height(v);
        di->img_info.columns = video_get_width(v);
        di->video = v;
        di->vid_type = VT_VL;
        di->fps   = video_get_fps(v);
        di->frame = video_alloc_frame(v);
        di->image  = di->frame->data;
        di->image2 = NULL;
        return 0;
    }

    return 1;
}

int step(display_info *di) {
    ppp_frame *frame = di->frame;
    const int n_blocks = (di->img_info.rows*di->img_info.columns)/64;
    ppp_motion motions[n_blocks];    
    uint8_t *tmp;
    int result;

    switch (di->vid_type) {
    case VT_NONE:
        redraw(di);
        return 0;
    case VT_PPP:
        result = ppp_video_frame_read(di->file, n_blocks, frame, motions);
        if (result != 0) {
            if (!feof(di->file))
                fprintf(stderr, "error while reading from file\n");
            fclose(di->file);
            di->vid_type = VT_NONE;
            fprintf(stderr, "playback finished\n");
            return 0;
        }
 
        /*
         * Decode new frame into image2 (with image being the old image),
         * then swap image and image2.
         */
        decode_video_frame(frame, motions, &(di->img_info),
                           di->image, di->image2);
        tmp = di->image;
        di->image = di->image2;
        di->image2 = tmp;
        
        redraw(di);
        break;
    case VT_VL:
        if (video_get_next_frame(di->video, di->frame) != 0) {
            video_close(di->video);
            di->vid_type = VT_NONE;
            fprintf(stderr, "playback finished\n");
            return 0;
        }
        redraw(di);
        break;
    }
    return 1;
}

Uint32 timer(Uint32 interval, void * data) {
    display_info *di = (display_info*)data;
    if (step(di) == 0)
        exit(0);
    return interval;
}

void usage(const char *progname) {
    fprintf(stderr, "USAGE: %s [-s] FILE\n"
            "  FILE can be a (binary encoded) PGM image, a PPPI image,\n"
            "  a PPPV video or any video libavcodec can load.\n"
            "  Options:\n"
            "    -s   single step video\n",
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

    int frame_millis = lrint(1000 / di.fps);
    /* enable time if playing a video and we are not single stepping */
    if (!single_step && di.vid_type != VT_NONE)
        SDL_AddTimer(frame_millis, timer, &di);
    else
        step(&di);

    /* set key repeat interval (and delay) to match the frame rate */
    SDL_EnableKeyRepeat(SDL_DEFAULT_REPEAT_DELAY, frame_millis);

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
            case SDLK_SPACE:
            case SDLK_RETURN:
                if (single_step)
                    step(&di);
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
