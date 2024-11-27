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

bool load_file_to_bytes(const char *path, unsigned char **bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize);  // Allocate memory to hold the file data
    if (buffer == NULL) {
        fprintf(stderr, "%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file);  // Read the file into the buffer
    if (ferror(file)) {
        fprintf(stderr, "%s: read error: %s\n", __func__, strerror(errno));
        fclose(file);
        return false;
    }
    if (ret != (size_t)fileSize) {
        fprintf(stderr, "%s: unexpectedly reached end of file\n", __func__);
        fclose(file);
        return false;
    }
    fclose(file);  // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}