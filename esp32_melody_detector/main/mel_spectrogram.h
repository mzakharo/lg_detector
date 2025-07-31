#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include "config.h"
#include "audio_capture.h"
#include "esp_err.h"

// Mel spectrogram data structure
typedef struct {
    float data[N_MELS * MAX_SPECTROGRAM_WIDTH];  // Row-major order (freq x time)
    uint32_t timestamp;
} mel_spectrogram_t;

// Function declarations
esp_err_t mel_spectrogram_init(void);
esp_err_t mel_spectrogram_compute(const audio_window_t* audio_window, mel_spectrogram_t* spectrogram);
void mel_spectrogram_print_debug(const mel_spectrogram_t* spectrogram);

#endif // MEL_SPECTROGRAM_H
