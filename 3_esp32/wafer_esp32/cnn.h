#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// one convolution layer: channels in/out, input size, whether it pools, and its weights
typedef struct {
    int16_t C, O, H, W, pool;
    const int8_t *w;
    const int32_t *b;
    const int32_t *M;
    const uint8_t *S;
} cnn_layer_t;

// counts: the wafer as 2 x 56 x 56 bytes; a logit above 0 means that defect is present
void cnn_run(const uint8_t *counts, int32_t logits[8]);

#ifdef __cplusplus
}
#endif
