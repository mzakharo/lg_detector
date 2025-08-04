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

// DEBUG MODE: 0=normal, 1=microphone test, 2=spectrogram debug, 3=sine wave test, 4=dump audio
#define DEBUG_MODE 0
#include "test_vector.h"
// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"

// DSP annd FFT
#include "dsps_fft2r.h"
#include "dsps_wind.h"

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

    constexpr int kTensorArenaSize = 100 * 1024;
    //uint8_t tensor_arena[kTensorArenaSize];

    // NEW: I2S channel handle for the new driver
    i2s_chan_handle_t rx_chan;

    // Buffers for DSP (same as before)
    //float audio_buffer[WINDOW_SAMPLES];
    float fft_input[N_FFT];
    //float fft_output[N_FFT * 2];
    // REMOVED: double power_spectrum[FFT_BINS]; // This was causing accumulation - now using local arrays only
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

    // Global Hann window buffer
    float hann_window[N_FFT];
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
            .clk = PDM_CLK_PIN,
            .din = PDM_DATA_PIN,
            .invert_flags = {
                .clk_inv = true,
            },
        },
    };
        // Configure for mono recording (single microphone)
    //pdm_rx_cfg.slot_cfg.slot_mask = I2S_PDM_SLOT_LEFT;  // Use left slot
 

    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg));

    // 3. Enable the channel
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
    ESP_LOGI(TAG, "PDM microphone initialized successfully.");
}

// --- FIXED: Spectrogram Generation and Detection Task ---
void init_tflite();
// REVISED SIGNATURE: Passes a work buffer to avoid stack overflow
void generate_spectrogram_frame(const float* audio_frame, float* out_mel_spectrogram, float* fft_work_buffer);

#if DEBUG_MODE == 3
// --- DEBUG: Sine Wave Generation ---
void generate_sine_wave(float* buffer, int samples, float frequency, float amplitude) {
    for (int i = 0; i < samples; i++) {
        buffer[i] = amplitude * sin(2 * M_PI * frequency * i / SAMPLE_RATE);
    }
}
#endif

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
#elif DEBUG_MODE == 4
// --- DEBUG: Audio Dump Task (Buffered) ---
void debug_dump_audio_task(void* arg) {
    ESP_LOGI(TAG, "=== AUDIO DUMP DEBUG MODE (BUFFERED) ===");
    
    constexpr int DUMP_DURATION_S = 1;
    constexpr int DUMP_SAMPLES = SAMPLE_RATE * DUMP_DURATION_S;

    // Allocate a large buffer in PSRAM (or internal RAM if PSRAM is not available)
    int16_t* dump_buffer = (int16_t*)heap_caps_malloc(DUMP_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!dump_buffer) {
        ESP_LOGE(TAG, "Failed to allocate memory for audio dump buffer!");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Allocated %d bytes for dump buffer.", DUMP_SAMPLES * sizeof(int16_t));

    ESP_LOGI(TAG, "Recording %d seconds of audio...", DUMP_DURATION_S);

    int16_t* i2s_read_buffer;
    int samples_recorded = 0;

    // --- Step 1: Record audio into the large buffer ---
    while (samples_recorded < DUMP_SAMPLES) {
        if (xQueueReceive(filled_audio_queue, &i2s_read_buffer, portMAX_DELAY) == pdPASS) {
            int samples_to_copy = I2S_BUFFER_SIZE_BYTES / sizeof(int16_t);
            // Prevent buffer overflow if the last chunk is smaller
            if (samples_recorded + samples_to_copy > DUMP_SAMPLES) {
                samples_to_copy = DUMP_SAMPLES - samples_recorded;
            }
            
            memcpy(&dump_buffer[samples_recorded], i2s_read_buffer, samples_to_copy * sizeof(int16_t));
            samples_recorded += samples_to_copy;

            // Return the I2S buffer to the empty queue
            xQueueSend(empty_audio_queue, &i2s_read_buffer, 0);
        }
    }

    ESP_LOGI(TAG, "Recording complete. Now dumping %d samples...", samples_recorded);

    // --- Step 2: Dump the entire buffer to the console ---
    printf("--- START AUDIO DUMP ---\n");
    for (int i = 0; i < samples_recorded-1; i++) {
        printf("%d,", dump_buffer[i]);
    }
    printf("%d", dump_buffer[samples_recorded-1]);

    printf("--- END AUDIO DUMP ---\n");

    // --- Step 3: Clean up ---
    free(dump_buffer);
    ESP_LOGI(TAG, "Finished dumping audio.");
    ESP_LOGI(TAG, "Set DEBUG_MODE back to 0 and recompile for normal operation.");

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
    
    float* fft_work_buffer = (float*)heap_caps_malloc(2 * N_FFT * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
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

#if DEBUG_MODE == 3
// --- SINE WAVE GENERATOR TASK (replaces I2S reader for testing) ---
void sine_wave_generator_task(void* arg) {
    int16_t* audio_buffer;
    
    // Sine wave parameters for frequency sweep
    static uint32_t sample_counter = 0;
    const float amplitude = 0.3f; // 30% of full scale to avoid clipping
    const float sweep_duration_sec = 5.0f; // 5 second sweep cycle
    const float base_freq = 440.0f; // Start at 440 Hz (A4)
    const float max_freq = 1760.0f; // End at 1760 Hz (A6, 2 octaves higher)

    // CORRECTED: Phase accumulator for proper chirp generation
    static float phase = 0.0f;
    
    ESP_LOGI(TAG, "=== SINE WAVE GENERATOR STARTED ===");
    ESP_LOGI(TAG, "Generating frequency sweep: %.0f Hz to %.0f Hz over %.1f seconds", 
             base_freq, max_freq, sweep_duration_sec);
    ESP_LOGI(TAG, "This will create a clear spectrogram pattern showing frequency progression");
    
    while (true) {
        // Wait for an empty buffer from the queue
        if (xQueueReceive(empty_audio_queue, &audio_buffer, portMAX_DELAY) == pdPASS) {
            int samples_in_buffer = I2S_BUFFER_SIZE_BYTES / sizeof(int16_t);
            
            // Generate sine wave samples
            for (int i = 0; i < samples_in_buffer; i++) {
                // Calculate time in seconds
                float time_sec = (float)sample_counter / SAMPLE_RATE;
                
                // Calculate current frequency using a triangular sweep pattern
                float sweep_progress = fmodf(time_sec, sweep_duration_sec) / sweep_duration_sec;
                
                // Create a triangular wave for frequency (up then down)
                float freq_multiplier;
                if (sweep_progress < 0.5f) {
                    // First half: sweep up
                    freq_multiplier = sweep_progress * 2.0f;
                } else {
                    // Second half: sweep down
                    freq_multiplier = 2.0f - (sweep_progress * 2.0f);
                }
                
                float current_freq = base_freq + (max_freq - base_freq) * freq_multiplier;
                
                // Generate sine wave sample using the accumulated phase
                float sample_float = amplitude * sinf(phase);

                // Update phase for the next sample
                phase += 2.0f * M_PI * current_freq / SAMPLE_RATE;
                if (phase > 2.0f * M_PI) {
                    phase -= 2.0f * M_PI;
                }
                
                // Convert to 16-bit integer
                audio_buffer[i] = (int16_t)(sample_float * 32767.0f);
                
                sample_counter++;
            }
            
            // Log frequency every second for verification
            static uint32_t last_log_time = 0;
            uint32_t current_time_sec = sample_counter / SAMPLE_RATE;
            if (current_time_sec != last_log_time) {
                float time_sec = (float)sample_counter / SAMPLE_RATE;
                float sweep_progress = fmodf(time_sec, sweep_duration_sec) / sweep_duration_sec;
                float freq_multiplier;
                if (sweep_progress < 0.5f) {
                    freq_multiplier = sweep_progress * 2.0f;
                } else {
                    freq_multiplier = 2.0f - (sweep_progress * 2.0f);
                }
                float current_freq = base_freq + (max_freq - base_freq) * freq_multiplier;
                
                ESP_LOGI(TAG, "Time: %lus, Frequency: %.0f Hz, Sweep progress: %.1f%%", 
                         current_time_sec, current_freq, sweep_progress * 100.0f);
                last_log_time = current_time_sec;
            }
            
            // Send the filled buffer to the processing task
            if (xQueueSend(filled_audio_queue, &audio_buffer, portMAX_DELAY) != pdPASS) {
                ESP_LOGE(TAG, "Failed to send to filled audio queue");
            }
        }
        
        // Small delay to simulate I2S timing (not critical for testing)
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
#else
void i2s_reader_task(void* arg) {
    int16_t* i2s_read_buffer;

    while (true) {
        // Check queue status before reading
        UBaseType_t filled_queue_size = uxQueueMessagesWaiting(filled_audio_queue);
        if (filled_queue_size > 1) {
            //ESP_LOGW(TAG, "Reader task: Pipeline is lagging! %d buffers waiting.", filled_queue_size);
        }

        // Wait for an empty buffer from the queue
        if (xQueueReceive(empty_audio_queue, &i2s_read_buffer, portMAX_DELAY) == pdPASS) {
            size_t bytes_read = 0;
            esp_err_t err = i2s_channel_read(rx_chan, i2s_read_buffer, I2S_BUFFER_SIZE_BYTES, &bytes_read, portMAX_DELAY);
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "I2S read error: %s", esp_err_to_name(err));
                 // Return buffer to the empty queue on error
                xQueueSend(empty_audio_queue, &i2s_read_buffer, 0);
                continue; // Skip to next loop iteration
            }
            
            if (bytes_read > 0) {
                // Calculate RMS for power level
                int64_t sum_squares = 0;
                int samples_read = bytes_read / sizeof(int16_t);
                for (int i = 0; i < samples_read; i++) {
                    sum_squares += (int64_t)i2s_read_buffer[i] * i2s_read_buffer[i];
                }
                float rms = sqrtf((float)sum_squares / samples_read);
                //ESP_LOGI(TAG, "Mic Power (RMS): %.2f", rms);

                // Send the filled buffer to the processing task
                if (xQueueSend(filled_audio_queue, &i2s_read_buffer, 0) != pdPASS) { // Use 0 timeout
                    ESP_LOGE(TAG, "Filled audio queue is full! Dropping buffer.");
                    // If sending fails, we must return the buffer to prevent a leak
                    xQueueSend(empty_audio_queue, &i2s_read_buffer, 0);
                }
            } else {
                // If no bytes were read, return the buffer to the empty queue immediately
                xQueueSend(empty_audio_queue, &i2s_read_buffer, 0);
            }
        }
    }
}
#endif

void spectrogram_task(void* arg) {
    // --- DC Removal High-Pass Filter State ---
    constexpr float DC_FILTER_ALPHA = 0.995f;  // High-pass filter coefficient (cutoff ~50Hz at 16kHz)
    float dc_filter_prev_input = 0.0f;
    float dc_filter_prev_output = 0.0f;
    
    // --- Pre-emphasis Filter State ---
    constexpr float PRE_EMPHASIS_ALPHA = 0.97f;
    float previous_sample = 0.0f;

    // --- Streaming Spectrogram Buffers ---
    int16_t* i2s_read_buffer; // This will be a pointer received from the queue
    
    // Audio ring buffer for FFT frames - increased size to prevent overwrites
    constexpr int AUDIO_RING_SIZE = N_FFT * 8; // Much larger buffer to handle timing mismatches
    float* audio_ring_buffer = (float*)malloc(AUDIO_RING_SIZE * sizeof(float));
    int audio_ring_head = 0;
    int audio_ring_count = 0;
    
    // CRITICAL FIX: Add separate tracking for spectrogram frame extraction
    // This ensures proper temporal alignment of overlapping windows
    int spectrogram_extraction_head = 0;
    
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
    constexpr int STREAM_HOP_SAMPLES = N_FFT;
    int samples_since_last_fft = 0;

    ESP_LOGI(TAG, "Starting streaming spectrogram generation (hop=%d samples)", STREAM_HOP_SAMPLES);

    // --- Main Processing Loop ---
    int64_t last_inference_time_ms = 0;
    while (true) {
        // Check queue status before processing
        UBaseType_t empty_queue_size = uxQueueMessagesWaiting(empty_audio_queue);
        if (empty_queue_size == 0) {
            ESP_LOGW(TAG, "Spectrogram task: Starving for data! No empty buffers available.");
        }

        // 1. Wait for audio data from the reader task
        if (xQueueReceive(filled_audio_queue, &i2s_read_buffer, pdMS_TO_TICKS(100)) == pdPASS) {
            int samples_read = I2S_BUFFER_SIZE_BYTES / sizeof(int16_t);

            // 2. Add new samples to the audio ring buffer (with DC removal and pre-emphasis)
            for (int i = 0; i < samples_read; i++) {
                // Normalize the 16-bit sample to a float
                float current_sample = (float)i2s_read_buffer[i] / 32768.0f;
                
                // Apply DC removal high-pass filter first
                // y[n] = x[n] - x[n-1] + α * y[n-1]
                float dc_filtered_sample = current_sample - dc_filter_prev_input + DC_FILTER_ALPHA * dc_filter_prev_output;
                
                // Update DC filter state
                dc_filter_prev_input = current_sample;
                dc_filter_prev_output = dc_filtered_sample;
                
                // Apply pre-emphasis filter to boost high frequencies
                float filtered_sample = dc_filtered_sample - PRE_EMPHASIS_ALPHA * previous_sample;
                
                // Store the filtered sample in the ring buffer
                audio_ring_buffer[audio_ring_head] = current_sample;//filtered_sample;
                
                // Update the previous sample for the next iteration
                previous_sample = dc_filtered_sample;

                audio_ring_head = (audio_ring_head + 1) % AUDIO_RING_SIZE;
                
                if (audio_ring_count < AUDIO_RING_SIZE) {
                    audio_ring_count++;
                }
                
                samples_since_last_fft++;

                // 3. Check if we should generate a new spectrogram column
                if (audio_ring_count >= N_FFT && samples_since_last_fft >= STREAM_HOP_SAMPLES) {
                    samples_since_last_fft = 0;

                    // Extract a window of the most recent audio samples.
                    // This creates a scrolling spectrogram.
                    int start_idx = (audio_ring_head - N_FFT + AUDIO_RING_SIZE) % AUDIO_RING_SIZE;
                    for (int j = 0; j < N_FFT; j++) {
                        audio_frame_buffer[j] = audio_ring_buffer[(start_idx + j) % AUDIO_RING_SIZE];
                    }

                    // Generate one spectrogram column (keep as power values for now)
                    float* current_column = &spectrogram_buffer[spectrogram_col_head * N_MELS];
                    
                    generate_spectrogram_frame(audio_frame_buffer, current_column, fft_work_buffer);

                    
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
                            int64_t current_time_ms = esp_timer_get_time() / 1000;
                            if (current_time_ms - last_inference_time_ms > 500) {
                                last_inference_time_ms = current_time_ms;
                            // --- REVISED NORMALIZATION LOGIC ---
                            // 1. Reorder the spectrogram from the circular buffer into the linear model_input_buffer
                            // This creates a coherent snapshot in time for normalization.
                            for (int col = 0; col < SPECTROGRAM_WIDTH; col++) {
                                // FIXED: spectrogram_col_head points to the NEXT position to write
                                // So the oldest column is at spectrogram_col_head, newest is at (spectrogram_col_head - 1)
                                int src_col_idx = (spectrogram_col_head + col) % SPECTROGRAM_WIDTH;
                                memcpy(&model_input_buffer[col * N_MELS], &spectrogram_buffer[src_col_idx * N_MELS], N_MELS * sizeof(float));
                            }

                            // 2. Find the maximum power value *within this coherent snapshot*
                            float max_power_in_window = 0.0f;
                            for (int i = 0; i < N_MELS * SPECTROGRAM_WIDTH; i++) {
                                if (model_input_buffer[i] > max_power_in_window) {
                                    max_power_in_window = model_input_buffer[i];
                                }
                            }

                            // Prevent division by zero
                            if (max_power_in_window <= 0.0f) {
                                max_power_in_window = 1e-10f;
                            }

                            // 3. Convert the snapshot to dB using its own maximum as the reference
                            const float amin = 1e-10f;
                            const float top_db = 80.0f;
                            for (int i = 0; i < N_MELS * SPECTROGRAM_WIDTH; i++) {
                                float power_val = fmaxf(amin, model_input_buffer[i]);
                                float db_val = 10.0f * log10f(power_val / max_power_in_window);
                                
                                // Apply top_db clipping
                                if (db_val < -top_db) {
                                    db_val = -top_db;
                                }
                                model_input_buffer[i] = db_val;
                            }
#if 0
                            // --- DEBUG: Print the entire model input buffer for comparison ---
                            // CORRECTED: Output in [time, frequency] format to match actual memory layout
                            printf("--- C++ SPECTROGRAM START ---\\n");
                            for (int col = 0; col < SPECTROGRAM_WIDTH; col++) {
                                for (int mel = 0; mel < N_MELS; mel++) {
                                    printf("%.4f,", model_input_buffer[col * N_MELS + mel]);
                                }
                                printf("\\n");
                            }
                            printf("--- C++ SPECTROGRAM END ---\\n");
#endif
                            // --- Run Inference ---
                            memcpy(input->data.f, model_input_buffer, N_MELS * SPECTROGRAM_WIDTH * sizeof(float));
                            #if 1
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
                                ESP_LOGI(TAG, " Other: %.5f, Melody Prob: %.5f | Confidence: %.2f", output->data.f[0], melody_prob, confidence_score);

                                if (confidence_score >= CONFIDENCE_THRESHOLD) {
                                    ESP_LOGI(TAG, "*********************");
                                    ESP_LOGI(TAG, "*** MELODY DETECTED ***");
                                    ESP_LOGI(TAG, "*********************");
                                    is_in_cooldown = true;
                                    last_trigger_time_ms = esp_timer_get_time() / 1000;
                                    confidence_score = 0.0f;
                                }
                            }
                            #endif
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
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %s", esp_err_to_name(ret));
        return;
    }
    ESP_LOGI(TAG, "FFT tables initialized successfully");

    // Pre-calculate the Hann window
    dsps_wind_hann_f32(hann_window, N_FFT);
    ESP_LOGI(TAG, "Hann window pre-calculated.");

#if DEBUG_MODE == 0

#elif DEBUG_MODE == 1
    ESP_LOGI(TAG, "=== MICROPHONE DEBUG MODE ENABLED ===");
    ESP_LOGI(TAG, "TFLite initialization skipped");
#elif DEBUG_MODE == 2
    ESP_LOGI(TAG, "=== SPECTROGRAM DEBUG MODE ENABLED ===");
    ESP_LOGI(TAG, "TFLite initialization skipped");
#elif DEBUG_MODE == 3
    ESP_LOGI(TAG, "=== SINE WAVE DEBUG MODE ENABLED ===");
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
#elif DEBUG_MODE == 4
    // Audio dump debug mode
    ESP_LOGI(TAG, "TFLite initialization skipped for audio dump");
    xTaskCreatePinnedToCore(i2s_reader_task, "i2s_reader_task", 1024 * 4, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(debug_dump_audio_task, "debug_dump_audio_task", 1024 * 8, NULL, 5, NULL, 1); // Increased stack for malloc
#elif DEBUG_MODE == 3
    // Sine wave debug mode
    init_tflite();
    xTaskCreatePinnedToCore(sine_wave_generator_task, "sine_wave_generator_task", 1024 * 4, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(spectrogram_task, "spectrogram_task", 1024 * 8, NULL, 5, NULL, 1);
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
    
    static_resolver.AddConv2D();
    static_resolver.AddMaxPool2D();
    static_resolver.AddFullyConnected();
    static_resolver.AddReshape();
    static_resolver.AddSoftmax();
    static_resolver.AddQuantize();
    static_resolver.AddDequantize();
    static_resolver.AddShape();
    static_resolver.AddStridedSlice();
    static_resolver.AddPack();
    static_resolver.AddLogistic();
    static_resolver.AddRelu();

       // Allocate tensor arena in PSRAM if available, otherwise internal RAM
    uint8_t * tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM, trying internal RAM...");
        return;
    }
    ESP_LOGI(TAG, "Allocated %d bytes for tensor arena", kTensorArenaSize);


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
   
    
    // 1. Apply Hann window to filtered audio frame and convert to complex format
    // dsps_fft2r_fc32 expects complex input (interleaved real/imaginary pairs)
    for (int i = 0; i < N_FFT; i++) {
        fft_work_buffer[i * 2 + 0] = audio_frame[i] * hann_window[i]; // Real part
        fft_work_buffer[i * 2 + 1] = 0.0f;                            // Imaginary part (zero for real input)
    }

    // 2. Perform FFT using ESP-DSP library
    // dsps_fft2r_fc32 expects complex input data (interleaved real/imag pairs)
    // Input: Complex data (real/imag interleaved) in the buffer
    // Output: Complex data (real/imag interleaved) in the same buffer
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

    
    // DEBUG: Log FFT output
    //ESP_LOGI(TAG, "FFT output: %.2e %.2e %.2e %.2e", fft_work_buffer[0], fft_work_buffer[1], fft_work_buffer[2], fft_work_buffer[3]);
       
    // Librosa's stft normalizes by the sum of the squared window.
    // For a Hann window, this sum is N_FFT * 3/8.
    // We apply this normalization to the power spectrum.
    //const float stft_normalization_factor = 1.0f / (float)(N_FFT * 3.0f / 8.0f);

    static float local_power_spectrum[FFT_BINS];
    memset(local_power_spectrum, 0, sizeof(local_power_spectrum));
    
    // 3. Calculate Power Spectrum from FFT output
    // This is |c|^2 for each complex bin c.
    // The result is the power of the signal at each frequency bin.
    // DC component (first bin)
    local_power_spectrum[0] = fft_work_buffer[0] * fft_work_buffer[0]; // Real part only
    
    // Positive frequencies (up to Nyquist)
    for (int i = 1; i < N_FFT / 2; i++) {
        float real = fft_work_buffer[i * 2];
        float imag = fft_work_buffer[i * 2 + 1];
        local_power_spectrum[i] = (real * real + imag * imag);
    }
    
    // Nyquist frequency (last bin)
    float nyquist_real = fft_work_buffer[N_FFT]; // Real part only
    local_power_spectrum[N_FFT / 2] = (nyquist_real * nyquist_real);


 
    // 4. Apply Mel Filterbank with overflow protection
    for (int i = 0; i < N_MELS; i++) {
        float mel_value = 0.0; 
        for (int j = 0; j < FFT_BINS; j++) {
            mel_value += local_power_spectrum[j] * mel_filter_bank[j][i];
        }
        
        // Check for overflow and clamp to reasonable range
        if (mel_value > 1e6) {  // Much lower threshold to catch issues earlier
            ESP_LOGW(TAG, "Mel value %d overflow: %.2e, clamping", i, mel_value);
            out_mel_spectrogram[i] = 1e6f;
        } else if (mel_value < 0.0) {
            out_mel_spectrogram[i] = 0.0f;
        } else {
            out_mel_spectrogram[i] = mel_value;// local_power_spectrum[8*i] + local_power_spectrum[8*i+1] + local_power_spectrum[8*i+2] + local_power_spectrum[8*i+3] + local_power_spectrum[8*i+4] + local_power_spectrum[8*i+5] + local_power_spectrum[8*i+6] + local_power_spectrum[8*i+7];
        }
    } 
}
