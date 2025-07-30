#include <iostream>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_dsp.h"
#include "esp_timer.h"
#include <cmath>

// NEW: Include the new I2S driver headers
#include "driver/i2s_pdm.h"
#include "driver/gpio.h" // For GPIO configuration
#include "esp_heap_caps.h" // For heap_caps_malloc

// DEBUG MODE: 0=normal, 1=microphone test, 2=spectrogram debug
#define DEBUG_MODE 0

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"

// DSP annd FFT
#include "dsps_fft2r.h"

// Local includes
#include "mel_filters.h"

// Include the model data
extern const unsigned char ___lg_sound_model_tflite[];
extern const int ___lg_sound_model_tflite_len;

// --- CONFIGURATION (MUST MATCH PYTHON SCRIPT) ---
static const char* TAG = "LG_DETECTOR";

// Audio settings
constexpr int SAMPLE_RATE = 16000;
constexpr float WINDOW_DURATION_S = 1.5f;
constexpr float HOP_DURATION_S = 0.5f;
constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
constexpr int I2S_BUFFER_SIZE_BYTES = 4096;

// Spectrogram settings
constexpr int N_FFT = 1024;
constexpr int N_MELS = MEL_FILTER_BANK_N_MELS;
constexpr int SPECTROGRAM_WIDTH = 48; // Must match training config

// Calculated sizes
constexpr int WINDOW_SAMPLES = static_cast<int>(SAMPLE_RATE * WINDOW_DURATION_S);
constexpr int HOP_SAMPLES = static_cast<int>(SAMPLE_RATE * HOP_DURATION_S);
constexpr int FFT_BINS = N_FFT / 2 + 1;

// Confidence Accumulator Parameters (same as before)
constexpr float PREDICTION_THRESHOLD = 0.80f;
constexpr float CONFIDENCE_THRESHOLD = 0.95f;
constexpr float INCREMENT_AMOUNT = 0.15f;
constexpr float DECAY_RATE = 0.05f;
constexpr int COOLDOWN_SECONDS = 10;

// Globals
namespace {
    //tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    constexpr int kTensorArenaSize = 2* 1024 * 1024;
    //uint8_t tensor_arena[kTensorArenaSize];

    // NEW: I2S channel handle for the new driver
    i2s_chan_handle_t rx_chan;

    // Buffers for DSP (same as before)
    //float audio_buffer[WINDOW_SAMPLES];
    float fft_input[N_FFT];
    //float fft_output[N_FFT * 2];
    double power_spectrum[FFT_BINS]; // Use double for power spectrum to prevent overflow
    float mel_spectrogram[N_MELS];

    // Confidence state (same as before)
    float confidence_score = 0.0f;
    bool is_in_cooldown = false;
    int64_t last_trigger_time_ms = 0;

    // Inter-task communication
    QueueHandle_t filled_audio_queue;
    QueueHandle_t empty_audio_queue;

    // Statically allocated audio buffers
    constexpr int NUM_AUDIO_BUFFERS = 4;
    int16_t* audio_buffers[NUM_AUDIO_BUFFERS];
}
#define PDM_CLK_PIN     (GPIO_NUM_42)  // PDM Clock
#define PDM_DATA_PIN    (GPIO_NUM_41)  // PDM Data

// --- I2S INITIALIZATION (COMPLETELY REWRITTEN FOR PDM) ---
void init_pdm_microphone() {
    // 1. Create a new I2S channel
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 3;
    chan_cfg.dma_frame_num = 300;
 
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    // 2. Configure the PDM RX mode
    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = (gpio_num_t)PDM_CLK_PIN,
            .din = (gpio_num_t)PDM_DATA_PIN,
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };
        // Configure for mono recording (single microphone)
    pdm_rx_cfg.slot_cfg.slot_mask = I2S_PDM_SLOT_LEFT;  // Use left slot
 

    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg));

    // 3. Enable the channel
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
    ESP_LOGI(TAG, "PDM microphone initialized successfully.");
}

// --- FIXED: Spectrogram Generation and Detection Task ---
void init_tflite();
// REVISED SIGNATURE: Passes a work buffer to avoid stack overflow
void generate_spectrogram_frame(const float* audio_frame, float* out_mel_spectrogram, float* fft_work_buffer);

#if DEBUG_MODE == 1
// --- DEBUG: Microphone Testing Task ---
void debug_microphone_task(void* arg) {
    ESP_LOGI(TAG, "=== MICROPHONE DEBUG MODE ===");
    ESP_LOGI(TAG, "Recording 5 seconds of audio for analysis...");
    
    int16_t* i2s_read_buffer;
    
    // Audio statistics
    int32_t min_val = INT32_MAX;
    int32_t max_val = INT32_MIN;
    int64_t sum_val = 0;
    int64_t sum_squares = 0;
    uint32_t total_samples = 0;
    uint32_t zero_samples = 0;
    uint32_t analysis_samples = 0; // Samples used for analysis (excluding first 50)
    
    // Recording parameters
    const uint32_t target_samples = SAMPLE_RATE * 5; // 5 seconds
    const uint32_t report_interval = SAMPLE_RATE; // Report every second
    uint32_t next_report = report_interval;
    const uint32_t ignore_samples = 50; // Ignore first 50 samples from analysis
    
    // Store first and last samples for pattern analysis
    constexpr int PATTERN_SAMPLES = 50;
    int16_t first_samples[PATTERN_SAMPLES];
    int16_t last_samples[PATTERN_SAMPLES];
    bool first_samples_captured = false;
    
    int64_t start_time = esp_timer_get_time();
    
    while (total_samples < target_samples) {
        if (xQueueReceive(filled_audio_queue, &i2s_read_buffer, portMAX_DELAY) == pdPASS) {
            int samples_in_buffer = I2S_BUFFER_SIZE_BYTES / sizeof(int16_t);
            
            for (int i = 0; i < samples_in_buffer && total_samples < target_samples; i++) {
                int16_t sample = i2s_read_buffer[i];
                
                // Capture first samples
                if (!first_samples_captured && total_samples < PATTERN_SAMPLES) {
                    first_samples[total_samples] = sample;
                    if (total_samples == PATTERN_SAMPLES - 1) {
                        first_samples_captured = true;
                    }
                }
                
                // Always capture last samples (rolling buffer)
                if (total_samples >= target_samples - PATTERN_SAMPLES) {
                    int idx = (total_samples - (target_samples - PATTERN_SAMPLES)) % PATTERN_SAMPLES;
                    last_samples[idx] = sample;
                }
                
                // Update statistics (exclude first 50 samples from analysis)
                if (total_samples >= ignore_samples) {
                    if (sample < min_val) min_val = sample;
                    if (sample > max_val) max_val = sample;
                    if (sample == 0) zero_samples++;
                    
                    sum_val += sample;
                    sum_squares += (int64_t)sample * sample;
                    analysis_samples++;
                }
                total_samples++;
                
                // Periodic reporting
                if (total_samples >= next_report) {
                    float elapsed_sec = (esp_timer_get_time() - start_time) / 1000000.0f;
                    float avg = (float)sum_val / total_samples;
                    float rms = sqrtf((float)sum_squares / total_samples);
                    
                    ESP_LOGI(TAG, "%.1fs: Samples=%lu, Current=%d, Min=%d, Max=%d, Avg=%.1f, RMS=%.1f", 
                             elapsed_sec, total_samples, sample, (int)min_val, (int)max_val, avg, rms);
                    
                    next_report += report_interval;
                }
            }
            
            // Return buffer to empty queue
            xQueueSend(empty_audio_queue, &i2s_read_buffer, 0);
        }
    }
    
    // Final analysis
    float elapsed_sec = (esp_timer_get_time() - start_time) / 1000000.0f;
    float avg = analysis_samples > 0 ? (float)sum_val / analysis_samples : 0.0f;
    float rms = analysis_samples > 0 ? sqrtf((float)sum_squares / analysis_samples) : 0.0f;
    float zero_percent = analysis_samples > 0 ? (float)zero_samples * 100.0f / analysis_samples : 0.0f;
    
    ESP_LOGI(TAG, "=== RECORDING COMPLETE ===");
    ESP_LOGI(TAG, "Recording time: %.2f seconds", elapsed_sec);
    ESP_LOGI(TAG, "Total samples: %lu (expected: %lu)", total_samples, target_samples);
    ESP_LOGI(TAG, "Sample rate achieved: %.1f Hz", total_samples / elapsed_sec);
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== AUDIO STATISTICS (excluding first 50 samples) ===");
    ESP_LOGI(TAG, "Analysis samples: %lu (total: %lu)", analysis_samples, total_samples);
    ESP_LOGI(TAG, "Min value: %d", (int)min_val);
    ESP_LOGI(TAG, "Max value: %d", (int)max_val);
    ESP_LOGI(TAG, "Range: %d", (int)(max_val - min_val));
    ESP_LOGI(TAG, "Average (DC offset): %.2f", avg);
    ESP_LOGI(TAG, "RMS (signal strength): %.2f", rms);
    ESP_LOGI(TAG, "Zero samples: %lu (%.2f%%)", zero_samples, zero_percent);
    ESP_LOGI(TAG, "");
    
    // Pattern analysis
    ESP_LOGI(TAG, "=== FIRST 50 SAMPLES ===");
    for (int i = 0; i < PATTERN_SAMPLES; i += 10) {
        ESP_LOGI(TAG, "Samples %2d-%2d: %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d", 
                 i, i+9,
                 first_samples[i], first_samples[i+1], first_samples[i+2], first_samples[i+3], first_samples[i+4],
                 first_samples[i+5], first_samples[i+6], first_samples[i+7], first_samples[i+8], first_samples[i+9]);
    }
    
    ESP_LOGI(TAG, "=== LAST 50 SAMPLES ===");
    for (int i = 0; i < PATTERN_SAMPLES; i += 10) {
        ESP_LOGI(TAG, "Samples %2d-%2d: %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d", 
                 i, i+9,
                 last_samples[i], last_samples[i+1], last_samples[i+2], last_samples[i+3], last_samples[i+4],
                 last_samples[i+5], last_samples[i+6], last_samples[i+7], last_samples[i+8], last_samples[i+9]);
    }
    
    // Analysis conclusions
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== ANALYSIS ===");
    if (zero_percent > 90.0f) {
        ESP_LOGI(TAG, "WARNING: >90%% zero samples - microphone may not be connected");
    } else if (zero_percent > 50.0f) {
        ESP_LOGI(TAG, "WARNING: >50%% zero samples - check microphone connection");
    } else {
        ESP_LOGI(TAG, "Good: Low zero sample percentage");
    }
    
    if (max_val - min_val < 100) {
        ESP_LOGI(TAG, "WARNING: Very low signal range - microphone may be faulty or no audio input");
    } else if (max_val - min_val > 60000) {
        ESP_LOGI(TAG, "WARNING: Signal may be clipping (range > 60000)");
    } else {
        ESP_LOGI(TAG, "Good: Signal range appears normal");
    }
    
    if (rms < 10.0f) {
        ESP_LOGI(TAG, "WARNING: Very low RMS - no significant audio detected");
    } else if (rms > 10000.0f) {
        ESP_LOGI(TAG, "WARNING: Very high RMS - signal may be too loud");
    } else {
        ESP_LOGI(TAG, "Good: RMS level appears reasonable");
    }
    
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== DEBUG COMPLETE ===");
    ESP_LOGI(TAG, "Set DEBUG_MICROPHONE to 0 and recompile for normal operation");
    
    // Keep task alive but idle
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
#elif DEBUG_MODE == 2
// --- DEBUG: Spectrogram Pipeline Debug Task ---
void debug_spectrogram_task(void* arg) {
    ESP_LOGI(TAG, "=== SPECTROGRAM DEBUG MODE ===");
    ESP_LOGI(TAG, "Analyzing spectrogram generation pipeline...");
    
    int16_t* i2s_read_buffer;
    
    // Allocate buffers for debugging
    constexpr int AUDIO_RING_SIZE = N_FFT * 2;
    float* audio_ring_buffer = (float*)malloc(AUDIO_RING_SIZE * sizeof(float));
    int audio_ring_head = 0;
    int audio_ring_count = 0;
    
    float* fft_work_buffer = (float*)heap_caps_malloc(N_FFT * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float* audio_frame_buffer = (float*)malloc(N_FFT * sizeof(float));
    float* debug_spectrogram = (float*)malloc(N_MELS * sizeof(float));
    
    if (!audio_ring_buffer || !fft_work_buffer || !audio_frame_buffer || !debug_spectrogram) {
        ESP_LOGE(TAG, "Failed to allocate debug buffers!");
        vTaskDelete(NULL);
        return;
    }
    
    memset(audio_ring_buffer, 0, AUDIO_RING_SIZE * sizeof(float));
    
    constexpr int STREAM_HOP_SAMPLES = N_FFT / 4;
    int samples_since_last_fft = 0;
    int spectrogram_count = 0;
    int64_t start_time = esp_timer_get_time();
    
    // Debug counters
    uint32_t total_audio_samples = 0;
    uint32_t total_spectrograms = 0;
    
    ESP_LOGI(TAG, "Starting debug analysis...");
    ESP_LOGI(TAG, "Will analyze first 10 spectrograms in detail");
    
    while (true) {
        if (xQueueReceive(filled_audio_queue, &i2s_read_buffer, portMAX_DELAY) == pdPASS) {
            int samples_read = I2S_BUFFER_SIZE_BYTES / sizeof(int16_t);
            
            // Log raw audio statistics every 1000 samples
            if (total_audio_samples % 1000 == 0 && total_audio_samples < 10000) {
                int16_t min_sample = INT16_MAX, max_sample = INT16_MIN;
                int64_t sum_sample = 0;
                
                for (int i = 0; i < samples_read; i++) {
                    int16_t sample = i2s_read_buffer[i];
                    if (sample < min_sample) min_sample = sample;
                    if (sample > max_sample) max_sample = sample;
                    sum_sample += sample;
                }
                
                float avg_sample = (float)sum_sample / samples_read;
                ESP_LOGI(TAG, "Audio samples %lu-%lu: min=%d, max=%d, avg=%.1f, range=%d", 
                         total_audio_samples, total_audio_samples + samples_read - 1,
                         min_sample, max_sample, avg_sample, max_sample - min_sample);
            }
            
            // Process audio samples
            for (int i = 0; i < samples_read; i++) {
                // Convert and add to ring buffer
                float normalized_sample = (float)i2s_read_buffer[i] / 32768.0f;
                audio_ring_buffer[audio_ring_head] = normalized_sample;
                audio_ring_head = (audio_ring_head + 1) % AUDIO_RING_SIZE;
                
                if (audio_ring_count < AUDIO_RING_SIZE) {
                    audio_ring_count++;
                }
                
                samples_since_last_fft++;
                total_audio_samples++;
                
                // Generate spectrogram when ready
                if (audio_ring_count >= N_FFT && samples_since_last_fft >= STREAM_HOP_SAMPLES) {
                    samples_since_last_fft = 0;
                    
                    // Extract audio frame
                    int start_idx = (audio_ring_head - N_FFT + AUDIO_RING_SIZE) % AUDIO_RING_SIZE;
                    for (int j = 0; j < N_FFT; j++) {
                        audio_frame_buffer[j] = audio_ring_buffer[(start_idx + j) % AUDIO_RING_SIZE];
                    }
                    
                    // Debug audio frame statistics (first 10 spectrograms only)
                    if (total_spectrograms < 10) {
                        float min_audio = 1.0f, max_audio = -1.0f, sum_audio = 0.0f;
                        for (int j = 0; j < N_FFT; j++) {
                            float val = audio_frame_buffer[j];
                            if (val < min_audio) min_audio = val;
                            if (val > max_audio) max_audio = val;
                            sum_audio += val;
                        }
                        float avg_audio = sum_audio / N_FFT;
                        
                        ESP_LOGI(TAG, "Spectrogram %lu - Audio frame: min=%d, max=%d, avg=%d, range=%d", 
                                 total_spectrograms, (int)(min_audio*10000), (int)(max_audio*10000), 
                                 (int)(avg_audio*10000), (int)((max_audio - min_audio)*10000));
                    }
                    
                    // Generate spectrogram
                    generate_spectrogram_frame(audio_frame_buffer, debug_spectrogram, fft_work_buffer);
                    
                    // Debug spectrogram statistics (first 10 spectrograms only)
                    if (total_spectrograms < 10) {
                        float min_spec = 1000.0f, max_spec = -1000.0f, sum_spec = 0.0f;
                        int zero_count = 0;
                        
                        for (int j = 0; j < N_MELS; j++) {
                            float val = debug_spectrogram[j];
                            if (val < min_spec) min_spec = val;
                            if (val > max_spec) max_spec = val;
                            sum_spec += val;
                            if (val == -60.0f) zero_count++; // Floor value
                        }
                        float avg_spec = sum_spec / N_MELS;
                        
                        ESP_LOGI(TAG, "Spectrogram %lu - Mel values: min=%d, max=%d, avg=%d, zeros=%d/%d", 
                                 total_spectrograms, (int)(min_spec*100), (int)(max_spec*100), (int)(avg_spec*100), zero_count, N_MELS);
                        
                        // Show first 10 mel values as integers (scaled by 100)
                        ESP_LOGI(TAG, "First 10 mel values: %d %d %d %d %d %d %d %d %d %d",
                                 (int)(debug_spectrogram[0]*100), (int)(debug_spectrogram[1]*100), (int)(debug_spectrogram[2]*100), 
                                 (int)(debug_spectrogram[3]*100), (int)(debug_spectrogram[4]*100), (int)(debug_spectrogram[5]*100), 
                                 (int)(debug_spectrogram[6]*100), (int)(debug_spectrogram[7]*100), (int)(debug_spectrogram[8]*100), 
                                 (int)(debug_spectrogram[9]*100));
                    }
                    
                    total_spectrograms++;
                    
                    // Summary every 50 spectrograms
                    if (total_spectrograms % 50 == 0) {
                        float elapsed_sec = (esp_timer_get_time() - start_time) / 1000000.0f;
                        ESP_LOGI(TAG, "Progress: %lu spectrograms generated in %.1fs (%.1f spec/sec)", 
                                 total_spectrograms, elapsed_sec, total_spectrograms / elapsed_sec);
                    }
                    
                    // Stop detailed analysis after 200 spectrograms
                    if (total_spectrograms >= 200) {
                        ESP_LOGI(TAG, "=== SPECTROGRAM DEBUG COMPLETE ===");
                        ESP_LOGI(TAG, "Generated %lu spectrograms from %lu audio samples", total_spectrograms, total_audio_samples);
                        ESP_LOGI(TAG, "If spectrograms are identical, the issue is in the pipeline");
                        ESP_LOGI(TAG, "If spectrograms vary but model output is constant, the issue is in the model");
                        
                        // Keep task alive but idle
                        while (true) {
                            vTaskDelay(pdMS_TO_TICKS(10000));
                        }
                    }
                }
            }
            
            xQueueSend(empty_audio_queue, &i2s_read_buffer, 0);
        }
    }
}
#endif

void i2s_reader_task(void* arg) {
    int16_t* i2s_read_buffer;

    while (true) {
        // Wait for an empty buffer from the queue
        if (xQueueReceive(empty_audio_queue, &i2s_read_buffer, portMAX_DELAY) == pdPASS) {
            size_t bytes_read = 0;
            i2s_channel_read(rx_chan, i2s_read_buffer, I2S_BUFFER_SIZE_BYTES, &bytes_read, portMAX_DELAY);
            
            ESP_LOGI(TAG, "bytes read %d", bytes_read);
            if (bytes_read > 0) {
                // Send the filled buffer to the processing task
                if (xQueueSend(filled_audio_queue, &i2s_read_buffer, portMAX_DELAY) != pdPASS) {
                    ESP_LOGE(TAG, "Failed to send to filled audio queue");
                }
            } else {
                // If no bytes were read, return the buffer to the empty queue
                xQueueSend(empty_audio_queue, &i2s_read_buffer, 0);
            }
        }
    }
}

void spectrogram_task(void* arg) {
    // --- Streaming Spectrogram Buffers ---
    int16_t* i2s_read_buffer; // This will be a pointer received from the queue
    
    // Audio ring buffer for FFT frames (only need N_FFT samples + some overlap)
    constexpr int AUDIO_RING_SIZE = N_FFT * 2; // Double buffer for overlap
    float* audio_ring_buffer = (float*)malloc(AUDIO_RING_SIZE * sizeof(float));
    int audio_ring_head = 0;
    int audio_ring_count = 0;
    
    // Spectrogram column buffer (circular buffer of columns)
    float* spectrogram_buffer = (float*)malloc(N_MELS * SPECTROGRAM_WIDTH * sizeof(float));
    int spectrogram_col_head = 0;
    int spectrogram_col_count = 0;
    
    // CRITICAL FIX: Allocate the FFT work buffer in internal RAM for the DSP library.
    // ESP-DSP real FFT requires 2*N_FFT buffer size for complex output
    float* fft_work_buffer = (float*)heap_caps_malloc(2 * N_FFT * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    // STACK OVERFLOW FIX: Allocate audio frame buffer on heap instead of stack
    float* audio_frame_buffer = (float*)malloc(N_FFT * sizeof(float));
    
    // STACK OVERFLOW FIX: Allocate model input buffer on heap instead of stack
    float* model_input_buffer = (float*)malloc(N_MELS * SPECTROGRAM_WIDTH * sizeof(float));

    if (!audio_ring_buffer || !spectrogram_buffer || !fft_work_buffer || !audio_frame_buffer || !model_input_buffer) {
        ESP_LOGE(TAG, "Failed to allocate one or more buffers!");
        vTaskDelete(NULL);
        return;
    }

    // Initialize buffers
    memset(audio_ring_buffer, 0, AUDIO_RING_SIZE * sizeof(float));
    memset(spectrogram_buffer, 0, N_MELS * SPECTROGRAM_WIDTH * sizeof(float));

    // --- TFLite Model Sanity Check ---
    if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Model is not Float32 type!");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Confirmed Float32 model.");

    // --- Log Input and Output Tensor Dimensions ---
    ESP_LOGI(TAG, "Input tensor dimensions:");
    for (int i = 0; i < input->dims->size; i++) {
        ESP_LOGI(TAG, "  dim %d: %d", i, input->dims->data[i]);
    }
    ESP_LOGI(TAG, "Output tensor dimensions:");
    for (int i = 0; i < output->dims->size; i++) {
        ESP_LOGI(TAG, "  dim %d: %d", i, output->dims->data[i]);
    }
    ESP_LOGI(TAG, "N_MELS = %d, SPECTROGRAM_WIDTH = %d", N_MELS, SPECTROGRAM_WIDTH);

    // Calculate hop size for streaming (smaller hops for more responsive detection)
    constexpr int STREAM_HOP_SAMPLES = N_FFT / 4; // 25% overlap, generates columns more frequently
    int samples_since_last_fft = 0;

    ESP_LOGI(TAG, "Starting streaming spectrogram generation (hop=%d samples)", STREAM_HOP_SAMPLES);

    // --- Main Processing Loop ---
    while (true) {
        // 1. Wait for audio data from the reader task
        if (xQueueReceive(filled_audio_queue, &i2s_read_buffer, portMAX_DELAY) == pdPASS) {
            int samples_read = I2S_BUFFER_SIZE_BYTES / sizeof(int16_t);

            // 2. Add new samples to the audio ring buffer
            for (int i = 0; i < samples_read; i++) {
                // Convert and add sample to ring buffer
                audio_ring_buffer[audio_ring_head] = (float)i2s_read_buffer[i] / 32768.0f;
                audio_ring_head = (audio_ring_head + 1) % AUDIO_RING_SIZE;
                
                if (audio_ring_count < AUDIO_RING_SIZE) {
                    audio_ring_count++;
                }
                
                samples_since_last_fft++;

                // 3. Check if we should generate a new spectrogram column
                if (audio_ring_count >= N_FFT && samples_since_last_fft >= STREAM_HOP_SAMPLES) {
                    samples_since_last_fft = 0;

                    // Extract the latest N_FFT samples for FFT into heap-allocated buffer
                    int start_idx = (audio_ring_head - N_FFT + AUDIO_RING_SIZE) % AUDIO_RING_SIZE;
                    
                    for (int j = 0; j < N_FFT; j++) {
                        audio_frame_buffer[j] = audio_ring_buffer[(start_idx + j) % AUDIO_RING_SIZE];
                    }

                    // --- Pre-processing: DC Offset Removal and Normalization ---
                    float mean = 0.0f;
                    for (int j = 0; j < N_FFT; j++) {
                        mean += audio_frame_buffer[j];
                    }
                    mean /= N_FFT;
                    for (int j = 0; j < N_FFT; j++) {
                        audio_frame_buffer[j] -= mean;
                    }

                    // Generate one spectrogram column
                    float* current_column = &spectrogram_buffer[spectrogram_col_head * N_MELS];
                    generate_spectrogram_frame(audio_frame_buffer, current_column, fft_work_buffer);


                    ESP_LOGI(TAG, "First 10 mel values: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
                                 current_column[0], current_column[1], current_column[2], current_column[3], current_column[4],
                                 current_column[5], current_column[6], current_column[7], current_column[8], current_column[9]);


                    
                    // Update spectrogram buffer pointers
                    spectrogram_col_head = (spectrogram_col_head + 1) % SPECTROGRAM_WIDTH;
                    if (spectrogram_col_count < SPECTROGRAM_WIDTH) {
                        spectrogram_col_count++;
                    }

                    // 4. Run inference when we have a full spectrogram
                    if (spectrogram_col_count == SPECTROGRAM_WIDTH) {
                        // --- Cooldown Logic ---
                        if (is_in_cooldown && (esp_timer_get_time() / 1000 - last_trigger_time_ms > COOLDOWN_SECONDS * 1000)) {
                            is_in_cooldown = false;
                            ESP_LOGI(TAG, "Cooldown finished. Resuming detection.");
                        }

                        if (!is_in_cooldown) {
                            // Rearrange spectrogram buffer for model input (columns in chronological order)
                            for (int col = 0; col < SPECTROGRAM_WIDTH; col++) {
                                int src_col = (spectrogram_col_head + col) % SPECTROGRAM_WIDTH;
                                memcpy(&model_input_buffer[col * N_MELS], &spectrogram_buffer[src_col * N_MELS], N_MELS * sizeof(float));
                            }

                            // --- Run Inference ---
                            memcpy(input->data.f, model_input_buffer, N_MELS * SPECTROGRAM_WIDTH * sizeof(float));
                            if (interpreter->Invoke() != kTfLiteOk) {
                                ESP_LOGE(TAG, "Invoke failed.");
                            } else {
                                float melody_prob = output->data.f[1];

                                // --- Confidence Accumulator Logic ---
                                if (melody_prob > PREDICTION_THRESHOLD) {
                                    confidence_score += INCREMENT_AMOUNT;
                                    if (confidence_score > 1.0f) confidence_score = 1.0f;
                                } else {
                                    confidence_score -= DECAY_RATE;
                                    if (confidence_score < 0.0f) confidence_score = 0.0f;
                                }
                                ESP_LOGI(TAG, "Melody Prob: %.2f | Confidence: %.2f", melody_prob, confidence_score);

                                if (confidence_score >= CONFIDENCE_THRESHOLD) {
                                    ESP_LOGI(TAG, "*********************");
                                    ESP_LOGI(TAG, "*** MELODY DETECTED ***");
                                    ESP_LOGI(TAG, "*********************");
                                    is_in_cooldown = true;
                                    last_trigger_time_ms = esp_timer_get_time() / 1000;
                                    confidence_score = 0.0f;
                                }
                            }
                        }
                    }
                }
            }
            // Return the processed buffer to the empty queue
            xQueueSend(empty_audio_queue, &i2s_read_buffer, 0);
        }
    }
}


extern "C" void app_main() {
    ESP_LOGI(TAG, "Initializing FFT tables for N_FFT=%d...", N_FFT);
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, N_FFT);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %s", esp_err_to_name(ret));
        return;
    }
    ESP_LOGI(TAG, "FFT tables initialized successfully");

#if DEBUG_MODE == 0

#elif DEBUG_MODE == 1
    ESP_LOGI(TAG, "=== MICROPHONE DEBUG MODE ENABLED ===");
    ESP_LOGI(TAG, "TFLite initialization skipped");
#elif DEBUG_MODE == 2
    ESP_LOGI(TAG, "=== SPECTROGRAM DEBUG MODE ENABLED ===");
    ESP_LOGI(TAG, "TFLite initialization skipped");
#endif

    // Create the audio queues
    filled_audio_queue = xQueueCreate(NUM_AUDIO_BUFFERS, sizeof(int16_t*));
    empty_audio_queue = xQueueCreate(NUM_AUDIO_BUFFERS, sizeof(int16_t*));

    // Allocate and populate the empty audio queue
    for (int i = 0; i < NUM_AUDIO_BUFFERS; i++) {
        audio_buffers[i] = (int16_t*)malloc(I2S_BUFFER_SIZE_BYTES);
        if (!audio_buffers[i]) {
            ESP_LOGE(TAG, "Failed to allocate audio buffer %d", i);
            return;
        }
        xQueueSend(empty_audio_queue, &audio_buffers[i], 0);
    }

    init_pdm_microphone();

#if DEBUG_MODE == 1
    // Microphone debug mode
    xTaskCreatePinnedToCore(i2s_reader_task, "i2s_reader_task", 1024 * 4, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(debug_microphone_task, "debug_microphone_task", 1024 * 4, NULL, 5, NULL, 1);
#elif DEBUG_MODE == 2
    // Spectrogram debug mode
    xTaskCreatePinnedToCore(i2s_reader_task, "i2s_reader_task", 1024 * 4, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(debug_spectrogram_task, "debug_spectrogram_task", 1024 * 6, NULL, 5, NULL, 1);
#else
    // Normal mode: Run full detection pipeline
    init_tflite();
    xTaskCreatePinnedToCore(i2s_reader_task, "i2s_reader_task", 1024 * 4, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(spectrogram_task, "spectrogram_task", 1024 * 8, NULL, 5, NULL, 1);
#endif
}



void init_tflite() {

    model = tflite::GetModel(___lg_sound_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG,"Model provided is schema version %lu not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // 3. Set up the OpResolver
    // This pulls in all operations, but is LARGE.
    // OLD: static tflite::AllOpsResolver resolver;

    // CORRECTED: Use MicroMutableOpResolver to only add the ops you need.
    // The template parameter <N> is the number of ops you plan to add.
    static tflite::MicroMutableOpResolver<12> static_resolver;
    
    // Add the specific operations your CNN uses.
    // Your model uses Conv2D, ReLU (part of Conv and Dense), MaxPool2D, Flatten, and FullyConnected (Dense).
    static_resolver.AddConv2D();
    static_resolver.AddMaxPool2D();
    static_resolver.AddFullyConnected(); // This is the 'Dense' layer
    static_resolver.AddReshape();       // Flatten often requires Reshape
    static_resolver.AddSoftmax();       // The final activation layer
    // ReLU is usually fused with the preceding op, but we can add it if needed.
    // The kernel for it is part of the fully_connected and conv kernels.
    static_resolver.AddQuantize();
    static_resolver.AddDequantize();
    static_resolver.AddShape();
    static_resolver.AddStridedSlice();
    static_resolver.AddPack();
    static_resolver.AddLogistic();

    uint8_t * tensor_arena = (uint8_t *) malloc(kTensorArenaSize);
    if (tensor_arena == NULL)
    {
        ESP_LOGE(TAG, "TFLITE ARENA ERROR");
        return;
    }

    // 4. Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
        model, static_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLite model failed.");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI(TAG, "TFLite model initialized.");
}

// --- FIXED: generate_spectrogram_frame() function ---
// REVISED to accept a work buffer, preventing stack overflow.
void generate_spectrogram_frame(const float* audio_frame, float* out_mel_spectrogram, float* fft_work_buffer) {
    // 1. Copy audio frame to the heap-allocated work buffer
    memcpy(fft_work_buffer, audio_frame, N_FFT * sizeof(float));

    // 2. Perform FFT using ESP-DSP library
    
    // CRITICAL FIX: ESP-DSP real FFT requires a buffer of size 2*N_FFT for complex output
    // We need to zero-pad the second half for the imaginary components
    for (int i = N_FFT; i < N_FFT * 2; i++) {
        fft_work_buffer[i] = 0.0f;
    }
    
    // Perform the real-to-complex FFT
    esp_err_t fft_result = dsps_fft2r_fc32(fft_work_buffer, N_FFT);
    
    
    if (fft_result != ESP_OK) {
        ESP_LOGE(TAG, "FFT failed with error %d", fft_result);
        // Set all outputs to silence if FFT fails
        for (int i = 0; i < N_MELS; i++) {
            out_mel_spectrogram[i] = -60.0f;
        }
        return;
    }
    
    // Apply bit reversal to get the correct frequency order
    dsps_bit_rev_fc32(fft_work_buffer, N_FFT);
    

    // 3. Calculate Power Spectrum from FFT output
    power_spectrum[0] = fft_work_buffer[0] * fft_work_buffer[0]; // DC
    for (int i = 1; i < N_FFT / 2; i++) {
        power_spectrum[i] = fft_work_buffer[i * 2] * fft_work_buffer[i * 2] + fft_work_buffer[i * 2 + 1] * fft_work_buffer[i * 2 + 1];
    }
    power_spectrum[N_FFT / 2] = fft_work_buffer[1] * fft_work_buffer[1]; // Nyquist
    

    // 4. Apply Mel Filterbank with overflow protection
    for (int i = 0; i < N_MELS; i++) {
        double mel_value = 0.0; // Use double precision to prevent overflow
        for (int j = 0; j < FFT_BINS; j++) {
            mel_value += (double)power_spectrum[j] * (double)mel_filter_bank[j][i];
        }
        
        // Check for overflow and clamp to reasonable range
        if (mel_value > 1e12) {
            out_mel_spectrogram[i] = 1e12f;
        } else if (mel_value < 0.0) {
            out_mel_spectrogram[i] = 0.0f;
        } else {
            out_mel_spectrogram[i] = (float)mel_value;
        }
    }
    

    // 5. Convert to dB scale (log), replicating librosa.power_to_db
    float max_val = 0.0f;
    for (int i = 0; i < N_MELS; i++) {
        if (out_mel_spectrogram[i] > max_val) {
            max_val = out_mel_spectrogram[i];
        }
    }


    const float amin = 1e-10f;
    const float top_db = 80.0f;

    for (int i = 0; i < N_MELS; i++) {
        float db_val = 10.0f * log10f(fmaxf(amin, out_mel_spectrogram[i]));
        db_val -= 10.0f * log10f(fmaxf(amin, max_val));
        out_mel_spectrogram[i] = fmaxf(db_val, db_val - top_db);
    }
    
}
