#include "cnn.h"

#include <string.h>

#include "model_data.h"

// two buffers for the feature maps: each layer reads one and writes the other
// every map has a 1-pixel border of zeros so the 3x3 window never goes off the edge
static uint8_t buf_a[MAP_BUF_BYTES], buf_b[MAP_BUF_BYTES];

// one 3x3 convolution output at (y, x), summed over all input channels
static int32_t conv_at(const uint8_t *in, const int8_t *w, int C, int Hp, int Wp, int y, int x) {
    int32_t acc = 0;
    for (int c = 0; c < C; c++)
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
                acc += in[(c * Hp + y + ky) * Wp + x + kx] * w[(c * 3 + ky) * 3 + kx];
    return acc;
}

// scale a sum back down to 0..255: (acc * m + rounding) >> s, then clip (the clip is also the ReLU)
static uint8_t requant(int32_t acc, int32_t m, int s) {
    int64_t p = ((int64_t)acc * m + ((int64_t)1 << (s - 1))) >> s;
    return p < 0 ? 0 : p > 255 ? 255 : (uint8_t)p;
}

// runs the whole network on one wafer and writes the 8 output scores into logits
void cnn_run(const uint8_t *counts, int32_t logits[8]) {
    uint8_t *in = buf_a, *out = buf_b;
    int Hp = IN_H + 2, Wp = IN_W + 2;
    // copy the wafer into the first buffer, inside the zero border
    memset(in, 0, sizeof buf_a);
    for (int c = 0; c < IN_C; c++)
        for (int y = 0; y < IN_H; y++)
            memcpy(in + (c * Hp + y + 1) * Wp + 1, counts + (c * IN_H + y) * IN_W, IN_W);

    for (int l = 0; l < N_LAYERS; l++) {
        const cnn_layer_t *L = &LAYERS[l];
        int oh = L->pool ? L->H / 2 : L->H, ow = L->pool ? L->W / 2 : L->W;
        int ohp = oh + 2, owp = ow + 2;
        memset(out, 0, sizeof buf_a);
        for (int o = 0; o < L->O; o++) {
            const int8_t *w = L->w + o * L->C * 9;
            for (int y = 0; y < oh; y++) {
                for (int x = 0; x < ow; x++) {
                    int32_t a;
                    if (L->pool) {  // 2x2 max pooling: keep the largest of the four convolution outputs
                        a = conv_at(in, w, L->C, Hp, Wp, 2 * y, 2 * x);
                        int32_t t = conv_at(in, w, L->C, Hp, Wp, 2 * y, 2 * x + 1);
                        if (t > a) a = t;
                        t = conv_at(in, w, L->C, Hp, Wp, 2 * y + 1, 2 * x);
                        if (t > a) a = t;
                        t = conv_at(in, w, L->C, Hp, Wp, 2 * y + 1, 2 * x + 1);
                        if (t > a) a = t;
                    } else {
                        a = conv_at(in, w, L->C, Hp, Wp, y, x);
                    }
                    out[(o * ohp + y + 1) * owp + x + 1] = requant(a + L->b[o], L->M[o], L->S[o]);
                }
            }
        }
        // this layer's output is the next layer's input
        uint8_t *t = in;
        in = out;
        out = t;
        Hp = ohp;
        Wp = owp;
    }

    // final layer: add up each channel's map, then a weighted sum for each of the 8 defect classes
    for (int k = 0; k < 8; k++) {
        int64_t z = HEAD_B[k];
        for (int c = 0; c < HEAD_C; c++) {
            int32_t sum = 0;
            for (int y = 0; y < LAST_H; y++)
                for (int x = 0; x < LAST_W; x++) sum += in[(c * Hp + y + 1) * Wp + x + 1];
            z += (int64_t)sum * HEAD_W[k * HEAD_C + c];
        }
        logits[k] = (int32_t)z;
    }
}
