#ifndef CONFIDENCE_TRACKER_H
#define CONFIDENCE_TRACKER_H

#include "config.h"
#include "esp_err.h"
#include <stdbool.h>

// Confidence tracker state
typedef struct {
    float confidence_score;
    bool is_in_cooldown;
    uint32_t last_trigger_time;
    uint32_t cooldown_duration_ticks;
} confidence_tracker_t;

// Function declarations
esp_err_t confidence_tracker_init(confidence_tracker_t* tracker);
bool confidence_tracker_update(confidence_tracker_t* tracker, float melody_probability);
void confidence_tracker_reset(confidence_tracker_t* tracker);

#endif // CONFIDENCE_TRACKER_H
