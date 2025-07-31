#include "tflite_inference.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <string.h>
#include <math.h>

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include the model data
#include "model_data.h"
#include "test_vector.h"
static const char *TAG = "TFLITE_INFERENCE";

// TensorFlow Lite Micro globals
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    constexpr int kTensorArenaSize = 100 * 1024;  // 2MB tensor arena
    uint8_t* tensor_arena = nullptr;
    
    bool is_initialized = false;
}

esp_err_t tflite_inference_init(void) {
    ESP_LOGI(TAG, "Initializing TensorFlow Lite Micro inference...");
    
    // Allocate tensor arena in PSRAM if available, otherwise internal RAM

    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena!");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Allocated %d bytes for tensor arena", kTensorArenaSize);
    
    // Load the model
    model = tflite::GetModel(___lg_sound_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %lu not supported (expected %d)", 
                model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_NOT_SUPPORTED;
    }
    ESP_LOGI(TAG, "Model loaded successfully (schema version %lu)", model->version());
    
    // Set up the operations resolver
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddLogistic();
    resolver.AddRelu();
    
    // Build the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate tensors!");
        return ESP_FAIL;
    }
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    if (!input || !output) {
        ESP_LOGE(TAG, "Failed to get input/output tensors!");
        return ESP_FAIL;
    }
    
    // Verify tensor types
    if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
        ESP_LOGE(TAG, "Model tensors are not Float32 type!");
        return ESP_ERR_NOT_SUPPORTED;
    }
    
    // Log tensor dimensions
    ESP_LOGI(TAG, "Input tensor dimensions:");
    for (int i = 0; i < input->dims->size; i++) {
        ESP_LOGI(TAG, "  dim %d: %d", i, input->dims->data[i]);
    }
    ESP_LOGI(TAG, "Output tensor dimensions:");
    for (int i = 0; i < output->dims->size; i++) {
        ESP_LOGI(TAG, "  dim %d: %d", i, output->dims->data[i]);
    }
    
    // Verify dimensions match our configuration
    int expected_input_size = N_MELS * MAX_SPECTROGRAM_WIDTH;
    int actual_input_size = 1;
    for (int i = 0; i < input->dims->size; i++) {
        actual_input_size *= input->dims->data[i];
    }
    
    if (actual_input_size != expected_input_size) {
        ESP_LOGE(TAG, "Input size mismatch: expected %d, got %d", expected_input_size, actual_input_size);
        return ESP_ERR_INVALID_SIZE;
    }
    
    int actual_output_size = 1;
    for (int i = 0; i < output->dims->size; i++) {
        actual_output_size *= output->dims->data[i];
    }
    
    if (actual_output_size != MODEL_OUTPUT_SIZE) {
        ESP_LOGE(TAG, "Output size mismatch: expected %d, got %d", MODEL_OUTPUT_SIZE, actual_output_size);
        return ESP_ERR_INVALID_SIZE;
    }
    
    is_initialized = true;
    
    ESP_LOGI(TAG, "TensorFlow Lite Micro initialized successfully");
    ESP_LOGI(TAG, "Model input size: %d", expected_input_size);
    ESP_LOGI(TAG, "Model output size: %d", actual_output_size);
    ESP_LOGI(TAG, "Tensor arena size: %d bytes", kTensorArenaSize);
    
    return ESP_OK;
}

esp_err_t tflite_inference_run(const mel_spectrogram_t* spectrogram, inference_result_t* result) {
    if (!spectrogram || !result || !is_initialized) {
        return ESP_ERR_INVALID_ARG;
    }
    
    result->timestamp = spectrogram->timestamp;
    
    // Convert spectrogram to dB scale (matching Python implementation)
    static float model_input_buffer[MODEL_INPUT_SIZE];
    
    // Find global maximum power for dB conversion
    float global_max_power = 0.0f;
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        if (spectrogram->data[i] > global_max_power) {
            global_max_power = spectrogram->data[i];
        }
    }
    
    // Prevent division by zero
    if (global_max_power <= 0.0f) {
        global_max_power = 1e-10f;
    }
    
    // Convert to dB using global maximum as reference (librosa.power_to_db behavior)
    const float amin = 1e-10f;
    const float top_db = 80.0f;
    
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        float power_val = fmaxf(amin, spectrogram->data[i]);
        
        // Convert to dB: 10 * log10(S / global_max_power)
        float db_val = 10.0f * log10f(power_val / global_max_power);
        
        // Apply top_db clipping (standard in librosa)
        if (db_val < -top_db) {
            db_val = -top_db;
        }
        
        model_input_buffer[i] = db_val;
    }
    
    // Copy input data to model
    memcpy(input->data.f, model_input_buffer, MODEL_INPUT_SIZE * sizeof(float));
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Model inference failed!");
        return ESP_FAIL;
    }
    
    // Copy output probabilities
    memcpy(result->probabilities, output->data.f, MODEL_OUTPUT_SIZE * sizeof(float));
    
    // Set convenience field
    result->lg_melody_probability = result->probabilities[LG_MELODY_CLASS_INDEX];
    
    #if DEBUG_LEVEL >= 2
    tflite_inference_print_debug(result);
    #endif
    
    return ESP_OK;
}

void tflite_inference_print_debug(const inference_result_t* result) {
    ESP_LOGI(TAG, "=== INFERENCE RESULT ===");
    ESP_LOGI(TAG, "Timestamp: %lu", result->timestamp);
    ESP_LOGI(TAG, "Probabilities:");
    ESP_LOGI(TAG, "  other_sounds: %.4f", result->probabilities[0]);
    ESP_LOGI(TAG, "  lg_melody:    %.4f", result->probabilities[1]);
    ESP_LOGI(TAG, "LG Melody Probability: %.4f", result->lg_melody_probability);
}
