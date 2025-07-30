#include <iostream>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_dsp.h"
#include "esp_timer.h"

// NEW: Include the new I2S driver headers
#include "driver/i2s_pdm.h"
#include "driver/gpio.h" // For GPIO configuration
#include "esp_heap_caps.h" // For heap_caps_malloc

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
    float power_spectrum[FFT_BINS];
    float mel_spectrogram[N_MELS];

    // Confidence state (same as before)
    float confidence_score = 0.0f;
    bool is_in_cooldown = false;
    int64_t last_trigger_time_ms = 0;

    // Inter-task communication
    QueueHandle_t audio_queue;
}
#define PDM_CLK_PIN     (GPIO_NUM_42)  // PDM Clock
#define PDM_DATA_PIN    (GPIO_NUM_41)  // PDM Dat

// --- I2S INITIALIZATION (COMPLETELY REWRITTEN FOR PDM) ---
void init_pdm_microphone() {
    // 1. Create a new I2S channel
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
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

void i2s_reader_task(void* arg) {
    int16_t* i2s_read_buffer = (int16_t*)malloc(I2S_BUFFER_SIZE_BYTES);
    if (!i2s_read_buffer) {
        ESP_LOGE(TAG, "Failed to allocate I2S read buffer!");
        vTaskDelete(NULL);
        return;
    }

    while (true) {
        size_t bytes_read = 0;
        i2s_channel_read(rx_chan, i2s_read_buffer, I2S_BUFFER_SIZE_BYTES, &bytes_read, portMAX_DELAY);
        if (bytes_read > 0) {
            // Send the buffer to the processing task
            if (xQueueSend(audio_queue, &i2s_read_buffer, portMAX_DELAY) != pdPASS) {
                ESP_LOGE(TAG, "Failed to send to audio queue");
            }
        }
    }
}

void spectrogram_task(void* arg) {
    // --- Buffers ---
    int16_t* i2s_read_buffer; // This will be a pointer received from the queue
    float* main_audio_buffer = (float*)malloc(WINDOW_SAMPLES * sizeof(float));
    float* model_input_buffer = (float*)malloc(N_MELS * SPECTROGRAM_WIDTH * sizeof(float));
    
    // CRITICAL FIX: Allocate the FFT work buffer in internal RAM for the DSP library.
    float* fft_work_buffer = (float*)heap_caps_malloc(N_FFT * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    if (!main_audio_buffer || !model_input_buffer || !fft_work_buffer) {
        ESP_LOGE(TAG, "Failed to allocate one or more buffers!");
        vTaskDelete(NULL);
        return;
    }

    int main_buffer_idx = 0;
    int spectrogram_col_idx = 0;

    // --- TFLite Model Sanity Check ---
    if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Model is not Float32 type!");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Confirmed Float32 model.");

    // --- Main Processing Loop ---
    while (true) {
        // 1. Wait for audio data from the reader task
        if (xQueueReceive(audio_queue, &i2s_read_buffer, portMAX_DELAY) == pdPASS) {
            int samples_read = I2S_BUFFER_SIZE_BYTES / sizeof(int16_t);

            // 2. Fill the main circular audio buffer
            for (int i = 0; i < samples_read; i++) {
                main_audio_buffer[main_buffer_idx] = (float)i2s_read_buffer[i] / 32768.0f;
                main_buffer_idx = (main_buffer_idx + 1);

                // 3. Check if we have enough data to generate a full spectrogram
                if (main_buffer_idx >= WINDOW_SAMPLES) {
                    ESP_LOGI(TAG, "Buffer full. Generating full spectrogram...");

                    // --- Cooldown Logic ---
                    if (is_in_cooldown && (esp_timer_get_time() / 1000 - last_trigger_time_ms > COOLDOWN_SECONDS * 1000)) {
                        is_in_cooldown = false;
                        ESP_LOGI(TAG, "Cooldown finished. Resuming detection.");
                    }

                    if (!is_in_cooldown) {
                        // --- FRAME-BY-FRAME SPECTROGRAM GENERATION ---
                        const int frame_hop = (WINDOW_SAMPLES - N_FFT) / (SPECTROGRAM_WIDTH - 1);

                        for (int frame_start = 0; (frame_start + N_FFT) <= WINDOW_SAMPLES && spectrogram_col_idx < SPECTROGRAM_WIDTH; frame_start += frame_hop) {
                            const float* audio_frame_ptr = &main_audio_buffer[frame_start];
                            generate_spectrogram_frame(audio_frame_ptr, &model_input_buffer[spectrogram_col_idx * N_MELS], fft_work_buffer);
                            spectrogram_col_idx++;
                        }
                        ESP_LOGI(TAG, "Generated %d columns.", spectrogram_col_idx);

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
                    // --- Slide the main buffer ---
                    memmove(main_audio_buffer, &main_audio_buffer[HOP_SAMPLES], (WINDOW_SAMPLES - HOP_SAMPLES) * sizeof(float));
                    main_buffer_idx = WINDOW_SAMPLES - HOP_SAMPLES;
                    spectrogram_col_idx = 0;
                }
            }
        }
    }
}


extern "C" void app_main() {
    ESP_LOGI(TAG, "Initializing FFT tables...");
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %s", esp_err_to_name(ret));
        return;
    }

    // Create the audio queue
    audio_queue = xQueueCreate(1, sizeof(int16_t*));

    init_pdm_microphone();
    init_tflite();

    // Create and start the tasks
    xTaskCreate(i2s_reader_task, "i2s_reader_task", 1024 * 4, NULL, 10, NULL);
    xTaskCreate(spectrogram_task, "spectrogram_task", 1024 * 8, NULL, 5, NULL);
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
    // 1. Copy audio frame to the heap-allocated work buffer and apply Hann window
    memcpy(fft_work_buffer, audio_frame, N_FFT * sizeof(float));
    dsps_wind_hann_f32(fft_work_buffer, N_FFT);

    // 2. Perform FFT using the work buffer
    dsps_fft2r_fc32(fft_work_buffer, N_FFT);
    dsps_bit_rev_fc32(fft_work_buffer, N_FFT);

    // 3. Calculate Power Spectrum for the frame
    // DC component
    power_spectrum[0] = fft_work_buffer[0] * fft_work_buffer[0];
    // Complex components
    for (int i = 1; i < N_FFT / 2; i++) {
        float real = fft_work_buffer[i * 2];
        float imag = fft_work_buffer[i * 2 + 1];
        power_spectrum[i] = (real * real) + (imag * imag);
    }
    // Nyquist component
    power_spectrum[N_FFT / 2] = fft_work_buffer[1] * fft_work_buffer[1];

    // 4. Apply Mel Filterbank
    for (int i = 0; i < N_MELS; i++) {
        out_mel_spectrogram[i] = 0.0f;
        for (int j = 0; j < FFT_BINS; j++) {
            out_mel_spectrogram[i] += power_spectrum[j] * mel_filter_bank[j][i];
        }
    }

    // 5. Convert to dB scale (log)
    for (int i = 0; i < N_MELS; i++) {
        float mel_val = out_mel_spectrogram[i];
        out_mel_spectrogram[i] = (mel_val > 1e-6) ? 10.0f * log10f(mel_val) : -60.0f;
    }
}
