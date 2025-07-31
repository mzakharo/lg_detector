#include "mel_spectrogram.h"
#include "esp_dsp.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <math.h>
#include <string.h>

static const char *TAG = "MEL_SPECTROGRAM";

// Dynamic buffers for DSP processing (allocated in PSRAM if available)
static float *fft_input = NULL;           // Complex input: real, imag, real, imag, ...
static float *power_spectrum = NULL;
static float *mel_filter_bank_data = NULL; // Flattened 2D array
static float *mel_energies = NULL;
static float *windowed_signal = NULL;

// Hanning window coefficients
static float *hanning_window = NULL;

// Helper macro for 2D array access
#define MEL_FILTER_BANK(mel_idx, bin_idx) mel_filter_bank_data[(mel_idx) * (N_FFT/2 + 1) + (bin_idx)]

// Helper function prototypes
static void generate_hanning_window(void);
static void generate_mel_filter_bank(void);
static float hz_to_mel(float hz);
static float mel_to_hz(float mel);
static void apply_window(const float* input, float* output, int length);
static void compute_power_spectrum(const float* fft_output, float* power_spec, int fft_size);
static void apply_mel_filters(const float* power_spec, float* mel_energies);
static void power_to_db(float* mel_energies, int n_mels);

esp_err_t mel_spectrogram_init(void) {
    ESP_LOGI(TAG, "Initializing mel spectrogram processor...");
    
    // Allocate dynamic buffers in PSRAM if available, otherwise internal RAM
    fft_input = (float*)heap_caps_malloc(N_FFT * 2 * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!fft_input) {
        ESP_LOGW(TAG, "Failed to allocate FFT input buffer in PSRAM, trying internal RAM...");
        fft_input = (float*)heap_caps_malloc(N_FFT * 2 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!fft_input) {
            ESP_LOGE(TAG, "Failed to allocate FFT input buffer!");
            return ESP_ERR_NO_MEM;
        }
    }
    
    power_spectrum = (float*)heap_caps_malloc((N_FFT/2 + 1) * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!power_spectrum) {
        power_spectrum = (float*)heap_caps_malloc((N_FFT/2 + 1) * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!power_spectrum) {
            ESP_LOGE(TAG, "Failed to allocate power spectrum buffer!");
            return ESP_ERR_NO_MEM;
        }
    }
    
    // Allocate mel filter bank as flattened 2D array
    size_t filter_bank_size = N_MELS * (N_FFT/2 + 1) * sizeof(float);
    mel_filter_bank_data = (float*)heap_caps_malloc(filter_bank_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!mel_filter_bank_data) {
        mel_filter_bank_data = (float*)heap_caps_malloc(filter_bank_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!mel_filter_bank_data) {
            ESP_LOGE(TAG, "Failed to allocate mel filter bank buffer!");
            return ESP_ERR_NO_MEM;
        }
    }
    
    mel_energies = (float*)heap_caps_malloc(N_MELS * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!mel_energies) {
        mel_energies = (float*)heap_caps_malloc(N_MELS * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!mel_energies) {
            ESP_LOGE(TAG, "Failed to allocate mel energies buffer!");
            return ESP_ERR_NO_MEM;
        }
    }
    
    windowed_signal = (float*)heap_caps_malloc(N_FFT * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!windowed_signal) {
        windowed_signal = (float*)heap_caps_malloc(N_FFT * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!windowed_signal) {
            ESP_LOGE(TAG, "Failed to allocate windowed signal buffer!");
            return ESP_ERR_NO_MEM;
        }
    }
    
    hanning_window = (float*)heap_caps_malloc(N_FFT * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!hanning_window) {
        hanning_window = (float*)heap_caps_malloc(N_FFT * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!hanning_window) {
            ESP_LOGE(TAG, "Failed to allocate Hanning window buffer!");
            return ESP_ERR_NO_MEM;
        }
    }
    
    ESP_LOGI(TAG, "Allocated dynamic buffers:");
    ESP_LOGI(TAG, "  FFT input: %d bytes", N_FFT * 2 * sizeof(float));
    ESP_LOGI(TAG, "  Power spectrum: %d bytes", (N_FFT/2 + 1) * sizeof(float));
    ESP_LOGI(TAG, "  Mel filter bank: %d bytes", (int)filter_bank_size);
    ESP_LOGI(TAG, "  Mel energies: %d bytes", N_MELS * sizeof(float));
    ESP_LOGI(TAG, "  Windowed signal: %d bytes", N_FFT * sizeof(float));
    ESP_LOGI(TAG, "  Hanning window: %d bytes", N_FFT * sizeof(float));
    
    // Initialize ESP-DSP
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, N_FFT);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize FFT: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Generate Hanning window
    generate_hanning_window();
    
    // Generate mel filter bank
    generate_mel_filter_bank();
    
    ESP_LOGI(TAG, "Mel spectrogram processor initialized");
    ESP_LOGI(TAG, "FFT size: %d", N_FFT);
    ESP_LOGI(TAG, "Mel bands: %d", N_MELS);
    ESP_LOGI(TAG, "Frequency range: 0 - %d Hz", SAMPLE_RATE/2);
    
    return ESP_OK;
}

esp_err_t mel_spectrogram_compute(const audio_window_t* audio_window, mel_spectrogram_t* spectrogram) {
    if (!audio_window || !spectrogram) {
        return ESP_ERR_INVALID_ARG;
    }
    
    spectrogram->timestamp = audio_window->timestamp;
    
    // Calculate hop size in samples for sliding window
    const int hop_samples = SAMPLE_RATE * HOP_DURATION_MS / 1000;  // 8000 samples
    const int window_samples = N_FFT;  // 1024 samples
    
    // Process overlapping windows to create spectrogram
    int time_step = 0;
    for (int start = 0; start + window_samples <= WINDOW_SAMPLES && time_step < MAX_SPECTROGRAM_WIDTH; 
         start += hop_samples, time_step++) {
        
        // Extract window from audio data
        memcpy(windowed_signal, &audio_window->samples[start], window_samples * sizeof(float));
        
        // Apply Hanning window
        apply_window(windowed_signal, windowed_signal, window_samples);
        
        // Prepare complex input for FFT (real, imag, real, imag, ...)
        memset(fft_input, 0, N_FFT * 2 * sizeof(float));
        for (int i = 0; i < window_samples; i++) {
            fft_input[i * 2] = windowed_signal[i];      // Real part
            fft_input[i * 2 + 1] = 0.0f;               // Imaginary part (zero)
        }
        
        // Compute FFT
        dsps_fft2r_fc32(fft_input, N_FFT);
        dsps_bit_rev_fc32(fft_input, N_FFT);
        
        // Compute power spectrum directly from complex FFT output
        compute_power_spectrum(fft_input, power_spectrum, N_FFT);
        
        // Apply mel filter bank
        apply_mel_filters(power_spectrum, mel_energies);
        
        // Convert to dB
        power_to_db(mel_energies, N_MELS);
        
        // Store in spectrogram (row-major order: freq x time)
        for (int mel_idx = 0; mel_idx < N_MELS; mel_idx++) {
            spectrogram->data[mel_idx * MAX_SPECTROGRAM_WIDTH + time_step] = mel_energies[mel_idx];
        }
    }
    
    // Pad remaining time steps with minimum dB value
    for (int mel_idx = 0; mel_idx < N_MELS; mel_idx++) {
        for (int t = time_step; t < MAX_SPECTROGRAM_WIDTH; t++) {
            spectrogram->data[mel_idx * MAX_SPECTROGRAM_WIDTH + t] = DB_MIN;
        }
    }
    
    #if DEBUG_LEVEL >= 3
    mel_spectrogram_print_debug(spectrogram);
    #endif
    
    return ESP_OK;
}

static void generate_hanning_window(void) {
    for (int i = 0; i < N_FFT; i++) {
        hanning_window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (N_FFT - 1)));
    }
}

static void generate_mel_filter_bank(void) {
    const float nyquist = SAMPLE_RATE / 2.0f;
    const int n_fft_bins = N_FFT / 2 + 1;
    
    // Create mel-spaced frequency points
    const float mel_low = hz_to_mel(0.0f);
    const float mel_high = hz_to_mel(nyquist);
    const float mel_step = (mel_high - mel_low) / (N_MELS + 1);
    
    float mel_points[N_MELS + 2];
    float hz_points[N_MELS + 2];
    int bin_points[N_MELS + 2];
    
    for (int i = 0; i < N_MELS + 2; i++) {
        mel_points[i] = mel_low + i * mel_step;
        hz_points[i] = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)floorf(hz_points[i] * N_FFT / SAMPLE_RATE);
        
        // Ensure bin is within valid range
        if (bin_points[i] >= n_fft_bins) {
            bin_points[i] = n_fft_bins - 1;
        }
    }
    
    // Initialize filter bank to zero
    memset(mel_filter_bank_data, 0, N_MELS * (N_FFT/2 + 1) * sizeof(float));
    
    // Create triangular filters
    for (int mel_idx = 0; mel_idx < N_MELS; mel_idx++) {
        int left_bin = bin_points[mel_idx];
        int center_bin = bin_points[mel_idx + 1];
        int right_bin = bin_points[mel_idx + 2];
        
        // Left slope (rising)
        for (int bin = left_bin; bin < center_bin; bin++) {
            if (center_bin > left_bin) {
                MEL_FILTER_BANK(mel_idx, bin) = (float)(bin - left_bin) / (center_bin - left_bin);
            }
        }
        
        // Right slope (falling)
        for (int bin = center_bin; bin < right_bin; bin++) {
            if (right_bin > center_bin) {
                MEL_FILTER_BANK(mel_idx, bin) = (float)(right_bin - bin) / (right_bin - center_bin);
            }
        }
    }
    
    ESP_LOGI(TAG, "Generated mel filter bank: %d filters, %d FFT bins", N_MELS, n_fft_bins);
}

static float hz_to_mel(float hz) {
    return MEL_SCALE_FACTOR * log10f(1.0f + hz / MEL_BREAK_FREQUENCY);
}

static float mel_to_hz(float mel) {
    return MEL_BREAK_FREQUENCY * (powf(10.0f, mel / MEL_SCALE_FACTOR) - 1.0f);
}

static void apply_window(const float* input, float* output, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * hanning_window[i];
    }
}

static void compute_power_spectrum(const float* fft_output, float* power_spec, int fft_size) {
    const int n_bins = fft_size / 2 + 1;
    
    // DC component
    power_spec[0] = fft_output[0] * fft_output[0];
    
    // Positive frequencies (complex pairs)
    for (int i = 1; i < n_bins - 1; i++) {
        float real = fft_output[2 * i];
        float imag = fft_output[2 * i + 1];
        power_spec[i] = real * real + imag * imag;
    }
    
    // Nyquist component (if fft_size is even)
    if (fft_size % 2 == 0) {
        power_spec[n_bins - 1] = fft_output[1] * fft_output[1];
    }
}

static void apply_mel_filters(const float* power_spec, float* mel_energies) {
    const int n_fft_bins = N_FFT / 2 + 1;
    
    for (int mel_idx = 0; mel_idx < N_MELS; mel_idx++) {
        mel_energies[mel_idx] = 0.0f;
        
        for (int bin = 0; bin < n_fft_bins; bin++) {
            mel_energies[mel_idx] += power_spec[bin] * MEL_FILTER_BANK(mel_idx, bin);
        }
        
        // Ensure minimum energy to avoid log(0)
        if (mel_energies[mel_idx] < 1e-10f) {
            mel_energies[mel_idx] = 1e-10f;
        }
    }
}

static void power_to_db(float* mel_energies, int n_mels) {
    for (int i = 0; i < n_mels; i++) {
        float db_value = 10.0f * log10f(mel_energies[i] / DB_REF_POWER);
        
        // Clamp to minimum dB value
        if (db_value < DB_MIN) {
            db_value = DB_MIN;
        }
        
        mel_energies[i] = db_value;
    }
}

void mel_spectrogram_print_debug(const mel_spectrogram_t* spectrogram) {
    ESP_LOGI(TAG, "=== MEL SPECTROGRAM DEBUG ===");
    ESP_LOGI(TAG, "Timestamp: %lu", spectrogram->timestamp);
    ESP_LOGI(TAG, "Shape: %d x %d (freq x time)", N_MELS, MAX_SPECTROGRAM_WIDTH);
    
    // Print first few values for verification
    ESP_LOGI(TAG, "First 5 values of first mel band:");
    for (int t = 0; t < 5 && t < MAX_SPECTROGRAM_WIDTH; t++) {
        ESP_LOGI(TAG, "  [0][%d] = %.4f", t, spectrogram->data[t]);
    }
    
    // Print spectrogram data in CSV format for comparison with Python
    ESP_LOGI(TAG, "=== C++ SPECTROGRAM DATA (CSV) ===");
    for (int mel_idx = 0; mel_idx < N_MELS; mel_idx++) {
        printf("%.4f", spectrogram->data[mel_idx * MAX_SPECTROGRAM_WIDTH]);
        for (int t = 1; t < MAX_SPECTROGRAM_WIDTH; t++) {
            printf(",%.4f", spectrogram->data[mel_idx * MAX_SPECTROGRAM_WIDTH + t]);
        }
        printf(",\n");
    }
    ESP_LOGI(TAG, "=== END SPECTROGRAM DATA ===");
}
