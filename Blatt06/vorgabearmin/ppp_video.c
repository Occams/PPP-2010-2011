#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ppp_video.h"
#include "frame_encoding.h"

static const char * const ppp_video_ident = "PPPVIDEO";

static const int motion_delta_max = 127;

ppp_motion motion_code(int deltaY, int deltaX) {
    if (abs(deltaY) <= motion_delta_max && abs(deltaX) <= motion_delta_max) {
        ppp_motion bm;
        bm.motionY = deltaY;
        bm.motionX = deltaX;
        return bm;
    }
    fprintf(stderr, "motion_code: error: delta too big.\n");
    return PPP_MOTION_INTRA;
}

bool is_intra(ppp_motion code) {
    return code.motionY == PPP_MOTION_INTRA.motionY &&
        code.motionX == PPP_MOTION_INTRA.motionX;
}

void decode_motion_code(ppp_motion code, int *deltaY, int *deltaX) {
    *deltaY = code.motionY;
    *deltaX = code.motionX;
}


FILE *ppp_video_write(const char *filename, const ppp_image_info *img_info,
                      const ppp_video_info *vid_info) {
    FILE *f;

    f = fopen(filename, "wb");
    if (f != NULL) {
        int len = strlen(ppp_video_ident);
        if (fwrite(ppp_video_ident, sizeof(char), len, f) == len &&
            fwrite(img_info, sizeof(*img_info), 1, f) == 1 &&
            fwrite(vid_info, sizeof(*vid_info), 1, f) == 1)
            return f;
        fclose(f);
    }
    return NULL;
}

static int write_block_motions(FILE *f, int n, const ppp_motion *motions) {
    return fwrite(motions, sizeof(*motions), n, f);
}

int ppp_video_frame_write(FILE *f, int n_blocks,
                          const ppp_frame *frame, const ppp_motion *motions) {
    size_t len = sizeof(*frame) + sizeof(frame->data[0])*(frame->length);
    if (write_block_motions(f, n_blocks, motions) != n_blocks)
        return -1;
    return fwrite(frame, len, 1, f) == 1 ? 0 : -1;
}

FILE *ppp_video_read(const char *filename, ppp_image_info *img_info,
                     ppp_video_info *vid_info) {
    FILE *f;

    f = fopen(filename, "rb");
    if (f != NULL) {
        ppp_video_header hdr;
        if (fread(&(hdr.ident), sizeof(hdr.ident), 1, f) == 1 &&
            fread(&(hdr.image_info), sizeof(hdr.image_info), 1, f) == 1 &&
            fread(&(hdr.video_info), sizeof(hdr.video_info), 1, f) == 1) {
            if (strncmp(hdr.ident, ppp_video_ident, sizeof(hdr.ident)) == 0) {
                *img_info = hdr.image_info;
                *vid_info = hdr.video_info;
                return f;
            }
        }
        fclose(f);
    }
    return NULL;
}

static int read_block_motions(FILE *f, int n, ppp_motion *motions) {
    return fread(motions, sizeof(*motions), n, f);
}

int ppp_video_frame_read(FILE *f, int n_blocks,
                         ppp_frame *frame, ppp_motion *motions) {
    if (read_block_motions(f, n_blocks, motions) != n_blocks)
        return -1;
    if (fread(&(frame->length), sizeof(frame->length), 1, f) != 1)
        return -1;
    
    if (fread(frame->data, sizeof(frame->data[0]), frame->length, f) != frame->length)
        return -1;

    return 0;
}
