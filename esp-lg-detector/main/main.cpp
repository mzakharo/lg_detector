#include <iostream>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/ringbuf.h"
#include "esp_log.h"
#include "esp_dsp.h"
#include "esp_timer.h"
#include <cmath>

// NEW: Include the new I2S driver headers
#include "driver/i2s_pdm.h"
#include "driver/gpio.h" // For GPIO configuration
#include "esp_heap_caps.h" // For heap_caps_malloc

// WiFi and MQTT includes
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "mqtt_client.h"

// DEBUG MODE: 0=normal, 1=sine wave test
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

// WiFi Configuration
#define WIFI_SSID "wireless"
#define WIFI_PASS ""

// MQTT Configuration
#define MQTT_BROKER_URI "mqtt://nas.local:1883"
#define MQTT_TOPIC "lg-detector/spectrogram"

// Audio settings
constexpr int SAMPLE_RATE = 16000;
constexpr float WINDOW_DURATION_S = 1.5f;
constexpr float HOP_DURATION_S = 0.5f;
constexpr i2s_port_t I2S_PORT = I2S_NUM_0;
constexpr int I2S_BUFFER_SIZE_BYTES = 1024;

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

    // Confidence state (same as before)
    float confidence_score = 0.0f;
    bool is_in_cooldown = false;
    int64_t last_trigger_time_ms = 0;

    // Inter-task communication - Ring buffer for audio streaming
    RingbufHandle_t audio_ring_buffer;
    
    // Ring buffer configuration
    constexpr int AUDIO_RING_BUFFER_SIZE = 8192; // 8KB ring buffer for audio data

    // Global Hann window buffer
    float hann_window[N_FFT];

    // WiFi and MQTT globals
    esp_mqtt_client_handle_t mqtt_client = nullptr;
    bool wifi_connected = false;
    bool mqtt_connected = false;
}
#define PDM_CLK_PIN     (GPIO_NUM_42)  // PDM Clock
#define PDM_DATA_PIN    (GPIO_NUM_41)  // PDM Data

// --- I2S INITIALIZATION (COMPLETELY REWRITTEN FOR PDM) ---
void init_pdm_microphone() {
    // 1. Create a new I2S channel
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    //chan_cfg.dma_desc_num = 3;
    //chan_cfg.dma_frame_num = 300;
 
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    // 2. Configure the PDM RX mode
    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = PDM_CLK_PIN,
            .din = PDM_DATA_PIN,
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg));

    // 3. Enable the channel
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));


    ESP_LOGI(TAG, "PDM microphone initialized successfully.");
}

// --- WiFi Event Handler ---
static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_connected = false;
        ESP_LOGI(TAG, "WiFi disconnected, attempting to reconnect...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "WiFi connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
        wifi_connected = true;
    }
}

// --- MQTT Event Handler ---
static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
   // esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    
    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected to broker");
            mqtt_connected = true;
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT disconnected from broker");
            mqtt_connected = false;
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGI(TAG, "MQTT error occurred");
            mqtt_connected = false;
            break;
        default:
            break;
    }
}

// --- WiFi Initialization ---
void init_wifi() {
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_got_ip));
    
    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char*)wifi_config.sta.password, WIFI_PASS);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_OPEN;  // Open network
    
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    ESP_LOGI(TAG, "WiFi initialization complete. Connecting to %s...", WIFI_SSID);
}

// --- MQTT Initialization ---
void init_mqtt() {
    esp_mqtt_client_config_t mqtt_cfg = {};
    mqtt_cfg.broker.address.uri = MQTT_BROKER_URI;
    
    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    if (mqtt_client == NULL) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        return;
    }
    
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(mqtt_client, (esp_mqtt_event_id_t)ESP_EVENT_ANY_ID, mqtt_event_handler, NULL));
    ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));
    
    ESP_LOGI(TAG, "MQTT client initialized and started");
}

// --- FIXED: Spectrogram Generation and Detection Task ---
void init_tflite();
// REVISED SIGNATURE: Passes a work buffer to avoid stack overflow
void generate_spectrogram_frame(const float* audio_frame, float* out_mel_spectrogram, float* fft_work_buffer);

#if DEBUG_MODE == 1
// --- SINE WAVE GENERATOR TASK (replaces I2S reader for testing) ---
void sine_wave_generator_task(void* arg) {
    // Allocate a local buffer for generating sine wave samples
    int16_t* audio_buffer = (int16_t*)malloc(I2S_BUFFER_SIZE_BYTES);
    if (!audio_buffer) {
        ESP_LOGE(TAG, "Failed to allocate audio buffer for sine wave generator");
        vTaskDelete(NULL);
        return;
    }
    
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
        
        // Send the audio data to the ring buffer
        if (xRingbufferSend(audio_ring_buffer, audio_buffer, I2S_BUFFER_SIZE_BYTES, portMAX_DELAY) != pdTRUE) {
            ESP_LOGE(TAG, "Failed to send audio data to ring buffer");
        }
        
        // Small delay to simulate I2S timing (not critical for testing)
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
#else
void i2s_reader_task(void* arg) {
    // Allocate a local buffer for reading I2S data
    int16_t* i2s_read_buffer = (int16_t*)malloc(I2S_BUFFER_SIZE_BYTES);
    if (!i2s_read_buffer) {
        ESP_LOGE(TAG, "Failed to allocate I2S read buffer");
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "I2S reader task started with ring buffer");

    while (true) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(rx_chan, i2s_read_buffer, I2S_BUFFER_SIZE_BYTES, &bytes_read, portMAX_DELAY);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "I2S read error: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(10)); // Brief delay before retry
            continue;
        }

        if (bytes_read == I2S_BUFFER_SIZE_BYTES) {            
            // Send the audio data directly to the ring buffer
            if (xRingbufferSend(audio_ring_buffer, i2s_read_buffer, I2S_BUFFER_SIZE_BYTES, pdMS_TO_TICKS(100)) != pdTRUE) {
                ESP_LOGW(TAG, "Ring buffer full! Dropping audio data.");
                // Continue reading to prevent I2S buffer overflow
            }
        } else {
            ESP_LOGW(TAG, "Incomplete I2S read: %d bytes (expected %d)", bytes_read, I2S_BUFFER_SIZE_BYTES);
        }
        
        // Small delay to prevent overwhelming the ring buffer
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
#endif

void spectrogram_task(void* arg) {
    // --- Simplified Spectrogram Buffers ---
    // Simple audio buffer to accumulate samples for FFT processing
    float* audio_accumulator = (float*)malloc(N_FFT * sizeof(float));
    int audio_accumulator_count = 0;
    
    // Spectrogram column buffer (circular buffer of columns)
    float* spectrogram_buffer = (float*)malloc(N_MELS * SPECTROGRAM_WIDTH * sizeof(float));
    int spectrogram_col_head = 0;
    int spectrogram_col_count = 0;
    
    // CRITICAL FIX: Allocate the FFT work buffer in internal RAM for the DSP library.
    // ESP-DSP real FFT requires N_FFT buffer size for real input.
    float* fft_work_buffer = (float*)heap_caps_aligned_alloc(16, N_FFT * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    // STACK OVERFLOW FIX: Allocate model input buffer on heap instead of stack
    float* model_input_buffer = (float*)malloc(N_MELS * SPECTROGRAM_WIDTH * sizeof(float));

    if (!audio_accumulator || !spectrogram_buffer || !fft_work_buffer || !model_input_buffer) {
        ESP_LOGE(TAG, "Failed to allocate one or more buffers!");
        vTaskDelete(NULL);
        return;
    }

    // Initialize buffers
    memset(audio_accumulator, 0, N_FFT * sizeof(float));
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
        // 1. Receive audio data from the ring buffer
        size_t item_size = 0;
        int16_t* received_data = (int16_t*)xRingbufferReceive(audio_ring_buffer, &item_size, pdMS_TO_TICKS(100));
        
        if (received_data != NULL) {
            int samples_read = item_size / sizeof(int16_t);

            // 2. Process samples directly - accumulate until we have enough for FFT
            for (int i = 0; i < samples_read; i++) {
                float current_sample = (float)received_data[i] / 32768.0f;
                           
                // Add sample to accumulator
                audio_accumulator[audio_accumulator_count] = current_sample;
                audio_accumulator_count++;
                samples_since_last_fft++;

                // 3. Check if we have enough samples for FFT processing
                if (audio_accumulator_count >= N_FFT && samples_since_last_fft >= STREAM_HOP_SAMPLES) {
                    samples_since_last_fft = 0;

                    // Generate one spectrogram column directly from accumulator
                    float* current_column = &spectrogram_buffer[spectrogram_col_head * N_MELS];
                    generate_spectrogram_frame(audio_accumulator, current_column, fft_work_buffer);
                    
                    // Shift accumulator by hop size (overlap processing)
                    memmove(audio_accumulator, &audio_accumulator[STREAM_HOP_SAMPLES], (N_FFT - STREAM_HOP_SAMPLES) * sizeof(float));
                    audio_accumulator_count = N_FFT - STREAM_HOP_SAMPLES;

                    
                    // Update spectrogram buffer pointers
                    spectrogram_col_head = (spectrogram_col_head + 1) % SPECTROGRAM_WIDTH;
                    if (spectrogram_col_count < SPECTROGRAM_WIDTH) {
                        spectrogram_col_count++;
                    }

                    // 4. Run inference when we have a full spectrogram
                    if (spectrogram_col_count == SPECTROGRAM_WIDTH) {
                        spectrogram_col_count -= SPECTROGRAM_WIDTH/4;
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
                            // --- MQTT Streaming: Publish spectrogram data before inference ---
                            if (mqtt_connected && mqtt_client != nullptr) {
                                int msg_id = esp_mqtt_client_publish(mqtt_client, 
                                                                   MQTT_TOPIC, 
                                                                   (char*)model_input_buffer, 
                                                                   N_MELS * SPECTROGRAM_WIDTH * sizeof(float), 
                                                                   0, 0);
                                if (msg_id == -1) {
                                    ESP_LOGW(TAG, "Failed to publish spectrogram data to MQTT");
                                } else {
                                    ESP_LOGI(TAG, "Published spectrogram data (%d bytes) to MQTT, msg_id=%d", 
                                             N_MELS * SPECTROGRAM_WIDTH * sizeof(float), msg_id);
                                }
                            } else {
                                ESP_LOGW(TAG, "MQTT not connected, skipping spectrogram publish");
                            }

                            // --- Run Inference ---
                            memcpy(input->data.f, model_input_buffer, N_MELS * SPECTROGRAM_WIDTH * sizeof(float));
                            #if 0
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
                                    //is_in_cooldown = true;
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
            
            // Return the item to the ring buffer
            vRingbufferReturnItem(audio_ring_buffer, (void*)received_data);
        }
    }
}


extern "C" void app_main() {
    ESP_LOGI(TAG, "Initializing FFT tables for N_FFT=%d...", N_FFT);
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, N_FFT/2); //CONFIG_DSP_MAX_FFT_SIZE
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT init failed: %s", esp_err_to_name(ret));
        return;
    }
    ret = dsps_fft4r_init_fc32(NULL, N_FFT/2);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT4R init failed: %s", esp_err_to_name(ret));
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

    // Create the audio ring buffer
    audio_ring_buffer = xRingbufferCreate(AUDIO_RING_BUFFER_SIZE, RINGBUF_TYPE_BYTEBUF);
    if (audio_ring_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to create audio ring buffer");
        return;
    }
    ESP_LOGI(TAG, "Audio ring buffer created successfully (%d bytes)", AUDIO_RING_BUFFER_SIZE);


    // Initialize WiFi and MQTT
    init_wifi();
    init_mqtt();

    init_pdm_microphone();

#if DEBUG_MODE == 1
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
    //uint8_t * tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint8_t * tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
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
    // 1. Apply Hann window and copy to work buffer for real FFT
    for (int i = 0; i < N_FFT; i++) {
        fft_work_buffer[i] = audio_frame[i] * hann_window[i];
    }

    // 2. Perform real FFT using ESP-DSP library
    dsps_fft2r_fc32(fft_work_buffer, N_FFT >> 1);
    
    // 3. Bit reverse for real FFT
    dsps_bit_rev2r_fc32(fft_work_buffer, N_FFT >> 1);

    // 4. Convert complex spectrum to real spectrum format
    dsps_cplx2real_fc32(fft_work_buffer, N_FFT >> 1);
    
    // 5. Calculate Power Spectrum from the real FFT output
    static float local_power_spectrum[FFT_BINS];

    // The output of dsps_cplx2real_fc32 is a packed spectrum.
    // DC component is in fft_work_buffer[0]
    // Nyquist component is in fft_work_buffer[1]
    // The rest are complex values.
    local_power_spectrum[0] = fft_work_buffer[0] * fft_work_buffer[0];
    local_power_spectrum[N_FFT / 2] = fft_work_buffer[1] * fft_work_buffer[1];

    for (int i = 1; i < N_FFT / 2; i++) {
        float real = fft_work_buffer[i * 2];
        float imag = fft_work_buffer[i * 2 + 1];
        local_power_spectrum[i] = (real * real + imag * imag);
    }

    // 6. Apply Mel Filterbank with overflow protection
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
            out_mel_spectrogram[i] = mel_value;
        }
    } 
}
