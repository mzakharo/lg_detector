#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "nvs_flash.h"

#include "config.h"
#include "audio_capture.h"
#include "mel_spectrogram.h"
#include "tflite_inference.h"
#include "confidence_tracker.h"

static const char *TAG = "MAIN";

// Global variables
static QueueHandle_t audio_queue = NULL;
static confidence_tracker_t confidence_tracker;
static bool system_running = false;

// Task handles
static TaskHandle_t processing_task_handle = NULL;


// Function prototypes
static void processing_task(void *pvParameters);
static void setup_led_indicator(void);
static void set_led_state(bool on);
static void print_system_info(void);

void app_main(void) {
    ESP_LOGI(TAG, "=== LG Melody Detector Starting ===");
    
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Print system information
    print_system_info();
    
    // Setup LED indicator
    setup_led_indicator();
    
    // Initialize all modules
    ESP_LOGI(TAG, "Initializing modules...");
    
    // Initialize mel spectrogram processor
    ESP_ERROR_CHECK(mel_spectrogram_init());
    
    // Initialize TensorFlow Lite inference
    ESP_ERROR_CHECK(tflite_inference_init());
    
    // Initialize confidence tracker
    ESP_ERROR_CHECK(confidence_tracker_init(&confidence_tracker));
    
    // Initialize audio capture
    ESP_ERROR_CHECK(audio_capture_init());
    
    // Create audio queue
    audio_queue = xQueueCreate(AUDIO_QUEUE_SIZE, sizeof(audio_window_t));
    if (audio_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create audio queue");
        return;
    }
    
    system_running = true;
    // Create processing task
    BaseType_t task_ret = xTaskCreate(
        processing_task,
        "processing",
        PROCESSING_TASK_STACK_SIZE,
        NULL,
        PROCESSING_TASK_PRIORITY,
        &processing_task_handle
    );
    
    if (task_ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create processing task");
        return;
    }
    
    // Start audio capture
    ESP_ERROR_CHECK(audio_capture_start(audio_queue));
    
    ESP_LOGI(TAG, "=== LG Melody Detector Started Successfully ===");
    ESP_LOGI(TAG, "Listening for LG washing machine melody...");
    
    // Main loop - just monitor system status
    while (system_running) {
        vTaskDelay(pdMS_TO_TICKS(5000));  // 5 second status updates
        
        #if DEBUG_LEVEL >= 1
        ESP_LOGI(TAG, "System running... Free heap: %lu bytes", esp_get_free_heap_size());
        #endif
    }
    
    // Cleanup (this code won't be reached in normal operation)
    ESP_LOGI(TAG, "Shutting down...");
    audio_capture_stop();
    
    if (processing_task_handle) {
        vTaskDelete(processing_task_handle);
    }
    
    if (audio_queue) {
        vQueueDelete(audio_queue);
    }
}

static void processing_task(void *pvParameters) {
    ESP_LOGI(TAG, "Processing task started");
    
    audio_window_t audio_window;
    mel_spectrogram_t spectrogram;
    inference_result_t inference_result;
    
    while (system_running) {
        // Wait for audio data
        if (xQueueReceive(audio_queue, &audio_window, portMAX_DELAY) == pdTRUE) {
            
            #if DEBUG_LEVEL >= 2
            ESP_LOGI(TAG, "Processing audio window (timestamp: %lu)", audio_window.timestamp);
            #endif
            
            // Compute mel spectrogram
            esp_err_t ret = mel_spectrogram_compute(&audio_window, &spectrogram);
            if (ret != ESP_OK) {
                ESP_LOGE(TAG, "Failed to compute mel spectrogram: %s", esp_err_to_name(ret));
                continue;
            }
            
            // Run inference
            ret = tflite_inference_run(&spectrogram, &inference_result);
            if (ret != ESP_OK) {
                ESP_LOGE(TAG, "Failed to run inference: %s", esp_err_to_name(ret));
                continue;
            }
            
            // Update confidence tracker
            bool melody_detected = confidence_tracker_update(&confidence_tracker, 
                                                           inference_result.lg_melody_probability);
            
            if (melody_detected) {
                ESP_LOGI(TAG, "🎵 LG WASHING MACHINE MELODY DETECTED! 🎵");
                
                // Turn on LED for 2 seconds
                set_led_state(true);
                vTaskDelay(pdMS_TO_TICKS(2000));
                set_led_state(false);
                
                // You can add additional actions here:
                // - Send notification
                // - Trigger relay
                // - Send data to cloud
                // - etc.
            }
            
            // Release the audio window back to the pool
            audio_capture_release_window(audio_window.samples);
            
            #if DEBUG_LEVEL >= 1
            static int process_count = 0;
            process_count++;
            if (process_count % 10 == 0) {  // Log every 10th processing cycle
                ESP_LOGI(TAG, "Processed %d windows, current probability: %.3f, confidence: %.3f", 
                        process_count, inference_result.lg_melody_probability, 
                        confidence_tracker.confidence_score);
            }
            #endif
        }
    }
    
    ESP_LOGI(TAG, "Processing task ended");
    vTaskDelete(NULL);
}

static void setup_led_indicator(void) {
    gpio_config_t io_conf = {
        .intr_type = GPIO_INTR_DISABLE,
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL << LED_GPIO),
        .pull_down_en = 0,
        .pull_up_en = 0,
    };
    gpio_config(&io_conf);
    
    // Turn off LED initially
    set_led_state(false);
    
    ESP_LOGI(TAG, "LED indicator configured on GPIO %d", LED_GPIO);
}

static void set_led_state(bool on) {
    gpio_set_level(LED_GPIO, on ? 1 : 0);
}

static void print_system_info(void) {
    ESP_LOGI(TAG, "=== SYSTEM INFORMATION ===");
    ESP_LOGI(TAG, "ESP-IDF Version: %s", esp_get_idf_version());
    ESP_LOGI(TAG, "Chip: %s", CONFIG_IDF_TARGET);
    ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "Minimum free heap: %lu bytes", esp_get_minimum_free_heap_size());
    
    ESP_LOGI(TAG, "=== AUDIO CONFIGURATION ===");
    ESP_LOGI(TAG, "Sample rate: %d Hz", SAMPLE_RATE);
    ESP_LOGI(TAG, "Window duration: %d ms (%d samples)", WINDOW_DURATION_MS, WINDOW_SAMPLES);
    ESP_LOGI(TAG, "Hop duration: %d ms (%d samples)", HOP_DURATION_MS, HOP_SAMPLES);
    ESP_LOGI(TAG, "FFT size: %d", N_FFT);
    ESP_LOGI(TAG, "Mel bands: %d", N_MELS);
    ESP_LOGI(TAG, "Max spectrogram width: %d", MAX_SPECTROGRAM_WIDTH);
    
    ESP_LOGI(TAG, "=== MODEL CONFIGURATION ===");
    ESP_LOGI(TAG, "Model input size: %d", MODEL_INPUT_SIZE);
    ESP_LOGI(TAG, "Model output size: %d", MODEL_OUTPUT_SIZE);
    ESP_LOGI(TAG, "LG melody class index: %d", LG_MELODY_CLASS_INDEX);
    
    ESP_LOGI(TAG, "=== DETECTION PARAMETERS ===");
    ESP_LOGI(TAG, "Prediction threshold: %.2f", PREDICTION_THRESHOLD);
    ESP_LOGI(TAG, "Confidence threshold: %.2f", CONFIDENCE_THRESHOLD);
    ESP_LOGI(TAG, "Increment amount: %.2f", INCREMENT_AMOUNT);
    ESP_LOGI(TAG, "Decay rate: %.2f", DECAY_RATE);
    ESP_LOGI(TAG, "Cooldown duration: %d seconds", COOLDOWN_SECONDS);
    
    ESP_LOGI(TAG, "=== HARDWARE CONFIGURATION ===");
    ESP_LOGI(TAG, "PDM CLK GPIO: %d", PDM_CLK_GPIO);
    ESP_LOGI(TAG, "PDM DATA GPIO: %d", PDM_DATA_GPIO);
    ESP_LOGI(TAG, "LED GPIO: %d", LED_GPIO);
    
    ESP_LOGI(TAG, "=== DEBUG LEVEL: %d ===", DEBUG_LEVEL);
    ESP_LOGI(TAG, "========================");
}
