#include "cnn.h"

static const int WAFER_BYTES = 2 * 56 * 56;  // one wafer: two 56x56 maps
static uint8_t wafer[WAFER_BYTES];

void setup() {
    Serial.setRxBufferSize(WAFER_BYTES + 64);  // the default 256-byte buffer would drop most of a wafer
    Serial.begin(115200);
    Serial.setTimeout(100);
}

void loop() {
    if (Serial.available() == 0) return;
    if (Serial.read() != 0xA5) return;  // 0xA5 marks the start of a wafer
    if (Serial.readBytes(wafer, WAFER_BYTES) != WAFER_BYTES) return;

    int32_t logits[8];
    // run the model and time it
    int64_t start = esp_timer_get_time();
    cnn_run(wafer, logits);
    uint32_t us = (uint32_t)(esp_timer_get_time() - start);

    // reply: 0x5A, which defects were found (one bit each), the 8 scores, and the time in microseconds
    uint8_t reply[38];
    reply[0] = 0x5A;
    reply[1] = 0;
    for (int k = 0; k < 8; k++) {
        if (logits[k] > 0) reply[1] |= 1 << k;
        memcpy(reply + 2 + 4 * k, &logits[k], 4);
    }
    memcpy(reply + 34, &us, 4);
    Serial.write(reply, sizeof reply);
    Serial.flush();
}
