#include "clip_utils.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

clip_image_u8 *make_clip_image_u8() { return new clip_image_u8(); }
clip_image_f32 *make_clip_image_f32() { return new clip_image_f32(); }
void clip_image_u8_free(clip_image_u8 *img) {
    if (img->data) {
        delete[] img->data;
    }
    delete img;
}
void clip_image_f32_free(clip_image_f32 *img) {
    if (img->data) {
        delete[] img->data;
    }
    delete img;
}

void build_clip_img_from_data(const stbi_uc *data, int nx, int ny, clip_image_u8 *img) {
    img->nx = nx;
    img->ny = ny;
    img->size = nx * ny * 3;
    img->data = new uint8_t[img->size]();
    memcpy(img->data, data, img->size);
}

bool clip_image_load_from_bytes(const unsigned char *bytes, size_t bytes_length, struct clip_image_u8 *img) {
    int nx, ny, nc;
    auto data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to decode image bytes\n", __func__);
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}
