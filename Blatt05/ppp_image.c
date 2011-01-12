#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ppp_image.h"

const char * const ppp_image_ident = "PPPIMAGE";

int ppp_image_write(const char *filename,
		    const ppp_image_info *info,
		    const ppp_frame *frame) {
    int result = 1;
    FILE *f = fopen(filename, "wb");
    
    if (f != NULL) {
        ppp_image_header hdr;
        size_t len;
        strncpy(hdr.ident, ppp_image_ident, sizeof(hdr.ident));
        hdr.image_info = *info;
        fwrite(&hdr, sizeof(hdr), 1, f);
        len = sizeof(*frame) + sizeof(frame->data[0])*(frame->length);
        fwrite(frame, len, 1, f);
        result = 0;
        fclose(f);
    }

    return result;
}

ppp_frame *ppp_image_read(const char *filename,
                          ppp_image_info *info) {
    void *data = NULL;
    FILE *f = fopen(filename, "rb");
    
    if (f != NULL) {
        ppp_image_header hdr;
        if (fread(&hdr, sizeof(hdr), 1, f) == 1) {
            if (strncmp(hdr.ident, ppp_image_ident, sizeof(hdr.ident)) == 0) {
                *info = hdr.image_info;
                data = ppp_frame_read(f);
            }
        }
        fclose(f);
    }
    
    return data;
}

ppp_frame *ppp_frame_alloc(int databytes) {
    ppp_frame *frame;
    size_t len = sizeof(ppp_frame) + sizeof(frame->data[0])*databytes;
    frame = (ppp_frame *)malloc(len);
    return frame;
}

ppp_frame *ppp_frame_read(FILE *f) {
    ppp_frame *frame;

    uint32_t length;
    if (fread(&length, sizeof(length), 1, f) != 1)
        return NULL;
    if (length <= 0)
        return NULL;

    frame = ppp_frame_alloc(length);
    if (frame != NULL) {
        if (fread(frame->data, sizeof(frame->data[0]), length, f) == length)
            return frame;
        free(frame);
    }

    return NULL;
}
