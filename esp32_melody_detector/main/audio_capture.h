#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include "config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_err.h"

// Audio data structure
typedef struct {
    float *samples;  // Pointer to samples (allocated separately)
    uint32_t timestamp;
    size_t sample_count;
} audio_window_t;

// Function declarations
esp_err_t audio_capture_init(void);
esp_err_t audio_capture_start(QueueHandle_t audio_queue);
esp_err_t audio_capture_stop(void);
void audio_capture_task(void *pvParameters);
void audio_capture_release_window(float* window_data);

#endif // AUDIO_CAPTURE_H
