#include "audio_capture.h"
#include "driver/i2s_pdm.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>
#include <math.h>

static const char *TAG = "AUDIO_CAPTURE";

// Global variables
static i2s_chan_handle_t rx_handle = NULL;
static TaskHandle_t audio_task_handle = NULL;
static QueueHandle_t audio_output_queue = NULL;
static bool is_running = false;

// Circular buffer for continuous audio capture (dynamically allocated)
static float *audio_buffer = NULL;
static size_t buffer_write_pos = 0;
static size_t buffer_read_pos = 0;

// Pool of audio windows allocated in PSRAM
#define AUDIO_WINDOW_POOL_SIZE 8
static float **audio_window_pool = NULL;  // Array of pointers to audio windows
static bool *audio_window_used = NULL;    // Track which windows are in use
static int next_window_index = 0;

esp_err_t audio_capture_init(void) {
    ESP_LOGI(TAG, "Initializing PDM microphone...");
    
    // Allocate audio buffer in PSRAM if available, otherwise internal RAM
    size_t buffer_size = AUDIO_BUFFER_SIZE * sizeof(float);
    audio_buffer =  NULL;//(float*)heap_caps_malloc(buffer_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!audio_buffer) {
        ESP_LOGW(TAG, "Failed to allocate audio buffer in PSRAM, trying internal RAM...");
        audio_buffer = (float*)heap_caps_malloc(buffer_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!audio_buffer) {
            ESP_LOGE(TAG, "Failed to allocate audio buffer!");
            return ESP_ERR_NO_MEM;
        }
    }
    ESP_LOGI(TAG, "Allocated audio buffer: %d bytes (%d samples)", (int)buffer_size, AUDIO_BUFFER_SIZE);
    
    // Allocate audio window pool
    audio_window_pool = (float**)heap_caps_malloc(AUDIO_WINDOW_POOL_SIZE * sizeof(float*), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!audio_window_pool) {
        ESP_LOGE(TAG, "Failed to allocate audio window pool array!");
        return ESP_ERR_NO_MEM;
    }
    
    audio_window_used = (bool*)heap_caps_malloc(AUDIO_WINDOW_POOL_SIZE * sizeof(bool), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!audio_window_used) {
        ESP_LOGE(TAG, "Failed to allocate audio window usage tracking!");
        return ESP_ERR_NO_MEM;
    }
    
    // Allocate individual audio windows in PSRAM
    size_t window_size = WINDOW_SAMPLES * sizeof(float);
    for (int i = 0; i < AUDIO_WINDOW_POOL_SIZE; i++) {
        audio_window_pool[i] = (float*)heap_caps_malloc(window_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!audio_window_pool[i]) {
            ESP_LOGW(TAG, "Failed to allocate window %d in PSRAM, trying internal RAM...", i);
            audio_window_pool[i] = (float*)heap_caps_malloc(window_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            if (!audio_window_pool[i]) {
                ESP_LOGE(TAG, "Failed to allocate audio window %d!", i);
                return ESP_ERR_NO_MEM;
            }
        }
        audio_window_used[i] = false;
    }
    ESP_LOGI(TAG, "Allocated %d audio windows: %d bytes each", AUDIO_WINDOW_POOL_SIZE, (int)window_size);
    
    // Initialize buffer positions
    buffer_write_pos = 0;
    buffer_read_pos = 0;
    next_window_index = 0;
    
    // Configure I2S for PDM reception
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));
    
    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = PDM_CLK_GPIO,
            .din = PDM_DATA_GPIO,
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };
    
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx_handle, &pdm_rx_cfg));
    
    ESP_LOGI(TAG, "PDM microphone initialized successfully");
    ESP_LOGI(TAG, "Sample rate: %d Hz", SAMPLE_RATE);
    ESP_LOGI(TAG, "Window size: %d samples (%.1f ms)", WINDOW_SAMPLES, (float)WINDOW_DURATION_MS);
    ESP_LOGI(TAG, "Hop size: %d samples (%.1f ms)", HOP_SAMPLES, (float)HOP_DURATION_MS);
    
    return ESP_OK;
}

esp_err_t audio_capture_start(QueueHandle_t audio_queue) {
    if (is_running) {
        ESP_LOGW(TAG, "Audio capture already running");
        return ESP_ERR_INVALID_STATE;
    }
    
    audio_output_queue = audio_queue;
    
    // Enable I2S channel
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
    
    is_running = true;
    // Create audio capture task
    BaseType_t ret = xTaskCreate(
        audio_capture_task,
        "audio_capture",
        AUDIO_CAPTURE_TASK_STACK_SIZE,
        NULL,
        AUDIO_CAPTURE_TASK_PRIORITY,
        &audio_task_handle
    );
    
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create audio capture task");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Audio capture started");
    
    return ESP_OK;
}

esp_err_t audio_capture_stop(void) {
    if (!is_running) {
        return ESP_OK;
    }
    
    is_running = false;
    
    // Delete task
    if (audio_task_handle) {
        vTaskDelete(audio_task_handle);
        audio_task_handle = NULL;
    }
    
    // Disable I2S channel
    if (rx_handle) {
        i2s_channel_disable(rx_handle);
    }
    
    ESP_LOGI(TAG, "Audio capture stopped");
    return ESP_OK;
}

static void convert_i2s_to_float(const int16_t* i2s_data, float* float_data, size_t samples) {
    const float scale = 1.0f / 32768.0f;  // Convert int16 to float [-1.0, 1.0]
    
    for (size_t i = 0; i < samples; i++) {
        float_data[i] = (float)i2s_data[i] * scale;
    }
}

static void add_to_circular_buffer(const float* new_samples, size_t count) {
    for (size_t i = 0; i < count; i++) {
        audio_buffer[buffer_write_pos] = new_samples[i];
        buffer_write_pos = (buffer_write_pos + 1) % AUDIO_BUFFER_SIZE;
        
        // Handle buffer overflow (overwrite old data)
        if (buffer_write_pos == buffer_read_pos) {
            buffer_read_pos = (buffer_read_pos + 1) % AUDIO_BUFFER_SIZE;
        }
    }
}

// Get an available audio window from the pool
static float* get_audio_window(void) {
    for (int i = 0; i < AUDIO_WINDOW_POOL_SIZE; i++) {
        int idx = (next_window_index + i) % AUDIO_WINDOW_POOL_SIZE;
        if (!audio_window_used[idx]) {
            audio_window_used[idx] = true;
            next_window_index = (idx + 1) % AUDIO_WINDOW_POOL_SIZE;
            return audio_window_pool[idx];
        }
    }
    return NULL;  // No available windows
}

// Release an audio window back to the pool
static void release_audio_window(float* window_data) {
    for (int i = 0; i < AUDIO_WINDOW_POOL_SIZE; i++) {
        if (audio_window_pool[i] == window_data) {
            audio_window_used[i] = false;
            return;
        }
    }
}

static bool extract_window_from_buffer(float* window_data) {
    size_t available_samples = 0;
    
    // Calculate available samples in circular buffer
    if (buffer_write_pos >= buffer_read_pos) {
        available_samples = buffer_write_pos - buffer_read_pos;
    } else {
        available_samples = AUDIO_BUFFER_SIZE - buffer_read_pos + buffer_write_pos;
    }
    
    if (available_samples < WINDOW_SAMPLES) {
        return false;  // Not enough samples
    }
    
    // Extract window from circular buffer
    for (size_t i = 0; i < WINDOW_SAMPLES; i++) {
        size_t pos = (buffer_read_pos + i) % AUDIO_BUFFER_SIZE;
        window_data[i] = audio_buffer[pos];
    }
    
    // Advance read position by hop size
    buffer_read_pos = (buffer_read_pos + HOP_SAMPLES) % AUDIO_BUFFER_SIZE;
    
    return true;
}

void audio_capture_task(void *pvParameters) {
    const size_t chunk_size = 512;  // Process in smaller chunks
    static int16_t i2s_buffer[512];
    static float float_buffer[512];
    size_t bytes_read = 0;
    uint32_t total_samples_read = 0;
    uint32_t windows_extracted = 0;
    
    ESP_LOGI(TAG, "Audio capture task started");
    
    while (is_running) {
        // Read audio data from PDM microphone
        esp_err_t ret = i2s_channel_read(rx_handle, i2s_buffer, 
                                       chunk_size * sizeof(int16_t), 
                                       &bytes_read, portMAX_DELAY);
        
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "I2S read failed: %s", esp_err_to_name(ret));
            continue;
        }
        
        if (bytes_read == 0) {
            #if DEBUG_LEVEL >= 2
            ESP_LOGW(TAG, "I2S read returned 0 bytes");
            #endif
            continue;
        }
        
        size_t samples_read = bytes_read / sizeof(int16_t);
        total_samples_read += samples_read;
        
        #if DEBUG_LEVEL >= 3
        if (total_samples_read % 8000 == 0) {  // Log every 0.5 seconds worth of samples
            ESP_LOGI(TAG, "I2S read: %d bytes (%d samples), total: %lu samples", 
                    (int)bytes_read, (int)samples_read, total_samples_read);
        }
        #endif
        
        // Convert I2S data to float
        convert_i2s_to_float(i2s_buffer, float_buffer, samples_read);
        
        // Add to circular buffer
        add_to_circular_buffer(float_buffer, samples_read);
        
        // Try to extract complete windows (may extract multiple windows per cycle)
        while (true) {
            float* window_buffer = get_audio_window();
            if (!window_buffer) {
                #if DEBUG_LEVEL >= 3
                ESP_LOGW(TAG, "No available audio windows in pool");
                #endif
                break;
            }
            
            if (!extract_window_from_buffer(window_buffer)) {
                // Not enough samples yet, release the window and break
                release_audio_window(window_buffer);
                break;
            }
            
            // Successfully extracted a window
            windows_extracted++;
            audio_window_t window;
            window.samples = window_buffer;
            window.timestamp = xTaskGetTickCount();
            window.sample_count = WINDOW_SAMPLES;
            
            // Send window to processing queue
            if (audio_output_queue) {
                if (xQueueSend(audio_output_queue, &window, 0) != pdTRUE) {
                    //ESP_LOGW(TAG, "Audio queue full, dropping window");
                    // Release the window back to pool if queue is full
                    release_audio_window(window_buffer);
                }
            } else {
                ESP_LOGW(TAG, "No audio output queue configured!");
                // Release the window if no queue
                release_audio_window(window_buffer);
            }
            
            #if DEBUG_LEVEL >= 2
            // Calculate RMS for debugging
            float rms = 0.0f;
            for (int i = 0; i < WINDOW_SAMPLES; i++) {
                rms += window.samples[i] * window.samples[i];
            }
            rms = sqrtf(rms / WINDOW_SAMPLES);
            //ESP_LOGI(TAG, "Window captured #%lu, RMS: %.4f", windows_extracted, rms);
            #endif
        }
        
        #if DEBUG_LEVEL >= 2
        // Periodic status update
        static uint32_t last_status_time = 0;
        uint32_t current_time = xTaskGetTickCount();
        if (current_time - last_status_time > pdMS_TO_TICKS(5000)) {  // Every 5 seconds
            size_t available_samples = 0;
            if (buffer_write_pos >= buffer_read_pos) {
                available_samples = buffer_write_pos - buffer_read_pos;
            } else {
                available_samples = AUDIO_BUFFER_SIZE - buffer_read_pos + buffer_write_pos;
            }
            //ESP_LOGI(TAG, "Status: %lu samples read, %lu windows extracted, %d samples in buffer", 
            //        total_samples_read, windows_extracted, (int)available_samples);
            //last_status_time = current_time;
        }
        #endif
        
        // Small delay to prevent task from hogging CPU
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    ESP_LOGI(TAG, "Audio capture task ended");
    vTaskDelete(NULL);
}

// Public function to release audio window back to pool
void audio_capture_release_window(float* window_data) {
    release_audio_window(window_data);
}
