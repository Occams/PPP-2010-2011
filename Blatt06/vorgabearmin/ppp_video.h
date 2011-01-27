#ifndef _PPP_VIDEO_H
#define _PPP_VIDEO_H

#include <stdbool.h>
#include <stdint.h>
#include "ppp_image.h"


/*
 * Describe the motion estimation for a macro block.  'motionY' and
 * 'motionX' describe the offset of the location that is used to
 * predict the current block, i.e., motionY = motionX = -1 means that
 * the current block is predicted from the 8x8 rectangle 1 pixel to
 * the left and 1 pixel up inthe previous frame.  (The viewer of the
 * video sees this as a motion to the right.)
 *
 * 'motionY' and 'motionX' must be in the range -127..127 to
 * to describe motion estimation for a predicted block.
 * For an intra-coded block, 'motionY' and 'motionX' must
 * both be set to -128 (see constant PPP_MOTION_INTRA below).
 */
typedef struct {
    int8_t motionY;
    int8_t motionX;
} ppp_motion;

static const ppp_motion PPP_MOTION_INTRA = { .motionY=-128, .motionX=-128 };

/*
 * Return a ppp_motion struct describing the given motion
 * (deltaY,deltaX).
 */
ppp_motion motion_code(int deltaY, int deltaX);


bool is_intra(ppp_motion motion);
void decode_motion_code(ppp_motion motion, int *deltaY, int *deltaX);



/*
 * Additional parameters of video files
 * (in addition to ppp_image_info).
 */ 
typedef struct _ppp_video_info {
    uint32_t fps;          /* number of frames per 1000 seconds */
} ppp_video_info;


typedef struct _ppp_video_header {
    char ident[8];    /* "PPPVIDEO" */
    ppp_image_info image_info;
    ppp_video_info video_info;
} ppp_video_header;

FILE *ppp_video_read(const char *filename,
                     ppp_image_info *img_info,
                     ppp_video_info *vid_info);
FILE *ppp_video_write(const char *filename,
                      const ppp_image_info *img_info,
                      const ppp_video_info *vid_info);
int ppp_video_frame_write(FILE *f, int n_blocks,
                          const ppp_frame *frame, const ppp_motion *motions);
int ppp_video_frame_read(FILE *f, int n_blocks,
                         ppp_frame *frame, ppp_motion *motions);


/*
 * Loading a video using FFMPEG.
 */

struct _video;
typedef struct _video video;

video *video_open(const char *filename);
void video_close(video *);
int video_get_width(const video *);
int video_get_height(const video *);
float video_get_fps(const video *);
ppp_frame *video_alloc_frame(const video *);
int video_get_next_frame(video *, ppp_frame *frame);

#endif
