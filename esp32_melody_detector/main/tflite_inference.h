#ifndef TFLITE_INFERENCE_H
#define TFLITE_INFERENCE_H

#include "config.h"
#include "mel_spectrogram.h"
#include "esp_err.h"

// Inference result structure
typedef struct {
    float probabilities[MODEL_OUTPUT_SIZE];  // [other_sounds, lg_melody]
    float lg_melody_probability;             // Convenience field for lg_melody class
    uint32_t timestamp;
} inference_result_t;

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations
esp_err_t tflite_inference_init(void);
esp_err_t tflite_inference_run(const mel_spectrogram_t* spectrogram, inference_result_t* result);
void tflite_inference_print_debug(const inference_result_t* result);

#ifdef __cplusplus
}
#endif

#endif // TFLITE_INFERENCE_H
