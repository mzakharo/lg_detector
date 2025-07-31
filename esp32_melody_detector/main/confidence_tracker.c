#include "confidence_tracker.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

static const char *TAG = "CONFIDENCE_TRACKER";

esp_err_t confidence_tracker_init(confidence_tracker_t* tracker) {
    if (!tracker) {
        return ESP_ERR_INVALID_ARG;
    }
    
    tracker->confidence_score = 0.0f;
    tracker->is_in_cooldown = false;
    tracker->last_trigger_time = 0;
    tracker->cooldown_duration_ticks = pdMS_TO_TICKS(COOLDOWN_SECONDS * 1000);
    
    ESP_LOGI(TAG, "Confidence tracker initialized");
    ESP_LOGI(TAG, "Prediction threshold: %.2f", PREDICTION_THRESHOLD);
    ESP_LOGI(TAG, "Confidence threshold: %.2f", CONFIDENCE_THRESHOLD);
    ESP_LOGI(TAG, "Increment amount: %.2f", INCREMENT_AMOUNT);
    ESP_LOGI(TAG, "Decay rate: %.2f", DECAY_RATE);
    ESP_LOGI(TAG, "Cooldown duration: %d seconds", COOLDOWN_SECONDS);
    
    return ESP_OK;
}

bool confidence_tracker_update(confidence_tracker_t* tracker, float melody_probability) {
    if (!tracker) {
        return false;
    }
    
    uint32_t current_time = xTaskGetTickCount();
    
    // Check if we're in cooldown period
    if (tracker->is_in_cooldown) {
        if (current_time - tracker->last_trigger_time >= tracker->cooldown_duration_ticks) {
            tracker->is_in_cooldown = false;
            ESP_LOGI(TAG, "Cooldown period ended");
        } else {
            // Still in cooldown, just apply decay
            tracker->confidence_score -= DECAY_RATE;
            if (tracker->confidence_score < 0.0f) {
                tracker->confidence_score = 0.0f;
            }
            return false;
        }
    }
    
    // Update confidence based on prediction
    if (melody_probability >= PREDICTION_THRESHOLD) {
        // High confidence prediction - increment
        tracker->confidence_score += INCREMENT_AMOUNT;
        if (tracker->confidence_score > 1.0f) {
            tracker->confidence_score = 1.0f;
        }
        
        #if DEBUG_LEVEL >= 2
        ESP_LOGI(TAG, "Melody detected (prob=%.3f), confidence=%.3f", 
                melody_probability, tracker->confidence_score);
        #endif
    } else {
        // Low confidence prediction - decay
        tracker->confidence_score -= DECAY_RATE;
        if (tracker->confidence_score < 0.0f) {
            tracker->confidence_score = 0.0f;
        }
        
        #if DEBUG_LEVEL >= 3
        ESP_LOGI(TAG, "No melody (prob=%.3f), confidence=%.3f", 
                melody_probability, tracker->confidence_score);
        #endif
    }
    
    // Check if confidence threshold is reached
    if (tracker->confidence_score >= CONFIDENCE_THRESHOLD) {
        ESP_LOGI(TAG, "*** LG MELODY DETECTED! *** (confidence=%.3f)", tracker->confidence_score);
        
        // Enter cooldown period
        tracker->is_in_cooldown = true;
        tracker->last_trigger_time = current_time;
        tracker->confidence_score = 0.0f;  // Reset confidence
        
        return true;  // Melody detected!
    }
    
    return false;  // No detection
}

void confidence_tracker_reset(confidence_tracker_t* tracker) {
    if (!tracker) {
        return;
    }
    
    tracker->confidence_score = 0.0f;
    tracker->is_in_cooldown = false;
    tracker->last_trigger_time = 0;
    
    ESP_LOGI(TAG, "Confidence tracker reset");
}
