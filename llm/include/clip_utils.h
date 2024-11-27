#include "common.h"

struct clip_image_u8 {
    int nx;
    int ny;
    uint8_t *data = NULL;
    size_t size;
};

struct clip_image_f32 {
    int nx;
    int ny;
    float *data = NULL;
    size_t size;
};

clip_image_u8 *make_clip_image_u8();
clip_image_f32 *make_clip_image_f32();
void clip_image_u8_free(clip_image_u8 *img);
void clip_image_f32_free(clip_image_f32 *img);

bool clip_image_load_from_bytes(const unsigned char *bytes, size_t bytes_length, struct clip_image_u8 *img);