#include <iostream>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_dsp.h"
#include "esp_timer.h"

// NEW: Include the new I2S driver headers
#include "driver/i2s_pdm.h"
#include "driver/gpio.h" // For GPIO configuration

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"

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

// init_tflite() and generate_spectrogram() remain exactly the same as the previous answer.
void init_tflite(); 
void generate_spectrogram(const float* audio_window);

// --- DETECTION TASK (UPDATED FOR FLOAT32 MODEL) ---
void detection_task(void* arg) {
    int16_t* i2s_read_buffer = (int16_t*)malloc(I2S_BUFFER_SIZE_BYTES);
    float* main_audio_buffer = (float*)malloc(WINDOW_SAMPLES * sizeof(float));
    if (i2s_read_buffer == NULL || main_audio_buffer == NULL) {
        ESP_LOGE(TAG, "buffer error");
        vTaskDelete(NULL);
        return;
    }
    // --- SOLUTION PART 1: Create a dedicated, writable buffer for the model input ---
    // This buffer will live in RAM and is guaranteed to be safe to write to.
    // Its size must exactly match the input tensor's size.
    float* model_input_buffer = (float*)malloc(N_MELS * SPECTROGRAM_WIDTH * sizeof(float));
    if (!model_input_buffer) {
        ESP_LOGE(TAG, "Failed to allocate memory for model input buffer!");
        vTaskDelete(NULL);
        return;
    }

    int main_buffer_idx = 0;

    // Verify tensor data types after allocation
    if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Model provided is not a float32 model!");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "Confirmed Float32 model input and output.");


    while (true) {
        size_t bytes_read = 0;
        i2s_channel_read(rx_chan, i2s_read_buffer, I2S_BUFFER_SIZE_BYTES, &bytes_read, portMAX_DELAY);
        int samples_read = bytes_read / sizeof(int16_t);
        for (int i = 0; i < samples_read; i++) {
            main_audio_buffer[main_buffer_idx] = (float)i2s_read_buffer[i] / 32768.0f;
            main_buffer_idx = (main_buffer_idx + 1) % WINDOW_SAMPLES;
        }

        if (main_buffer_idx < HOP_SAMPLES) {
            float processing_window[WINDOW_SAMPLES];
            for(int i = 0; i < WINDOW_SAMPLES; i++) {
                processing_window[i] = main_audio_buffer[(main_buffer_idx + i) % WINDOW_SAMPLES];
            }

            if (is_in_cooldown && (esp_timer_get_time() / 1000 - last_trigger_time_ms > COOLDOWN_SECONDS * 1000)) {
                is_in_cooldown = false;
                ESP_LOGI(TAG, "Cooldown finished. Resuming detection.");
            }

            if (!is_in_cooldown) {
                // Generate a single spectrogram from the current window
                //ESP_LOGI(TAG, "generate spectogram");
                generate_spectrogram(processing_window);
                //ESP_LOGI(TAG, "generated spectogram");

                // --- CHANGED: Populate Float32 Input Tensor ---
                // We directly copy the float values from our spectrogram into the input tensor.
                // This simplification replicates the same spectrogram column across the tensor width.
               // --- SOLUTION PART 2: Fill YOUR local RAM buffer, not the model's tensor directly ---
                for (int i = 0; i < SPECTROGRAM_WIDTH; ++i) {
                    for (int j = 0; j < N_MELS; ++j) {
                        // The destination is now our safe, local buffer
                        model_input_buffer[i * N_MELS + j] = mel_spectrogram[j];
                    }
                }

                // --- SOLUTION PART 3: Perform a single, safe, bulk copy ---
                // This copies the entire prepared input into the TFLite input tensor's data buffer.
                memcpy(input->data.f, model_input_buffer, N_MELS * SPECTROGRAM_WIDTH * sizeof(float));


                // Run Inference
                if (interpreter->Invoke() != kTfLiteOk) {
                    ESP_LOGE(TAG, "Invoke failed.");
                    continue;
                }

                // --- CHANGED: Read Float32 Output Tensor ---
                // Directly read the float value. No dequantization needed.
                // Adjust index [1] if your model's class order is different.
                float melody_prob = output->data.f[1];

                // --- Confidence logic remains identical ---
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


extern "C" void app_main() {
    init_pdm_microphone();
    init_tflite();
    xTaskCreate(detection_task, "detection_task", 1024 * 8, NULL, 5, NULL);
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

// --- CORRECTED generate_spectrogram() function ---
void generate_spectrogram(const float* audio_window) {
    // 1. Apply Hann window and copy to FFT buffer
    // Note: The dsps_wind_hann_f32 function is commented out as it modifies the input buffer
    // directly, which might be undesirable if the audio_window is used elsewhere.
    // A safer approach is to copy first, then apply the window.
    memcpy(fft_input, audio_window, WINDOW_SAMPLES * sizeof(float));

    // Zero pad the rest of the FFT input buffer if window is smaller than FFT
    for (int i = WINDOW_SAMPLES; i < N_FFT; ++i) {
        fft_input[i] = 0.0f;
    }
    
    // Apply window function to the copied data
    dsps_wind_hann_f32(fft_input, N_FFT);

    // 2. Perform FFT
    dsps_fft2r_fc32(fft_input, N_FFT);
    // Bit reverse
    dsps_bit_rev_fc32(fft_input, N_FFT);
    
    // --- CORRECTED POWER SPECTRUM CALCULATION ---
    // The previous loop was incorrect. This loop correctly handles the
    // packed format from dsps_fft2r_fc32.

    // Handle DC component (at index 0)
    power_spectrum[0] = fft_input[0] * fft_input[0];

    // Handle the complex components
    for (int i = 1; i < N_FFT / 2; i++) {
        float real = fft_input[i * 2];
        float imag = fft_input[i * 2 + 1];
        power_spectrum[i] = (real * real) + (imag * imag);
    }
    
    // Handle Nyquist component (at index 1)
    power_spectrum[N_FFT / 2] = fft_input[1] * fft_input[1];
    // --- END OF CORRECTION ---

    // 4. Apply Mel Filterbank
    for (int i = 0; i < N_MELS; i++) {
        mel_spectrogram[i] = 0.0f;
        for (int j = 0; j < FFT_BINS; j++) {
            mel_spectrogram[i] += power_spectrum[j] * mel_filter_bank[j][i];
        }
    }

    // 5. Convert to dB scale (log)
    for (int i = 0; i < N_MELS; i++) {
        // Use a small epsilon to avoid log(0)
        mel_spectrogram[i] = (mel_spectrogram[i] > 1e-6) ? 10.0f * log10f(mel_spectrogram[i]) : -60.0f;
    }
}