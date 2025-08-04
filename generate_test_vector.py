import librosa
import numpy as np
import tensorflow as tf

# --- Configuration (matching your config.h) ---
CONFIG = {
    "sample_rate": 16000,
    "window_duration": 1.5,  # seconds
    "hop_duration": 0.5,     # seconds
    "n_mels": 64,
    "n_fft": 1024,
    "max_spectrogram_width": 48
}

def process_audio_file(audio_path, config):
    """
    Process audio file to generate mel spectrogram matching C++ implementation
    """
    try:
        y, sr = librosa.load(audio_path, sr=config["sample_rate"])
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    window_samples = int(config["window_duration"] * config["sample_rate"])
    
    # Use first window (matching your C++ approach for single inference)
    if len(y) > window_samples:
        y = y[:window_samples]
    else:
        y = np.pad(y, (0, window_samples - len(y)), 'constant')

    # Generate Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=config["sample_rate"], 
        n_mels=config["n_mels"], 
        n_fft=config["n_fft"]
    )
    
    # Convert to dB (matching librosa.power_to_db with ref=np.max)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Standardize spectrogram width
    if log_spectrogram.shape[1] > config["max_spectrogram_width"]:
        log_spectrogram = log_spectrogram[:, :config["max_spectrogram_width"]]
    else:
        pad_width = config["max_spectrogram_width"] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    
    return log_spectrogram

def run_tflite_inference(model_path, input_data):
    """
    Run inference using TensorFlow Lite model
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input details:", input_details)
    print("Output details:", output_details)
    
    # Prepare input data - flatten and ensure correct shape
    input_shape = input_details[0]['shape']
    print(f"Expected input shape: {input_shape}")
    
    # Flatten the spectrogram (frequency x time -> single array)
    input_array = input_data.flatten().astype(np.float32)
    print(f"Input array shape: {input_array.shape}")
    print(f"Input data range: [{np.min(input_array):.2f}, {np.max(input_array):.2f}]")
    
    # Reshape to match model input
    if len(input_shape) == 4:  # Batch, Height, Width, Channels
        input_array = input_array.reshape(1, input_data.shape[0], input_data.shape[1], 1)
    elif len(input_shape) == 2:  # Batch, Features
        input_array = input_array.reshape(1, -1)
    else:
        input_array = input_array.reshape(input_shape)
    
    print(f"Reshaped input: {input_array.shape}")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get raw output
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    print('preditction', prediction)
    return prediction


def generate_c_header(spectrogram, expected_output, header_path):
    """
    Generate C header file with test vector data
    """
    # Flatten spectrogram in row-major order (frequency x time)
    flattened_spec = spectrogram.flatten()
    
    header_content = f"""#ifndef TEST_VECTOR_H
#define TEST_VECTOR_H

#include "config.h"

// Test vector generated from lg_washer.wav
// Input: {spectrogram.shape[0]} mel bands x {spectrogram.shape[1]} time steps = {len(flattened_spec)} values
// Expected output: [other_sounds, lg_melody] = [{expected_output[0]:.6f}, {expected_output[1]:.6f}]

// Test input spectrogram (flattened in row-major order: frequency x time)
static const float TEST_VECTOR_INPUT[MODEL_INPUT_SIZE] = {{
"""
    
    # Add spectrogram data (8 values per line for readability)
    for i in range(0, len(flattened_spec), 8):
        line_values = flattened_spec[i:i+8]
        line_str = "    " + ", ".join(f"{val:.6f}f" for val in line_values)
        if i + 8 < len(flattened_spec):
            line_str += ","
        header_content += line_str + "\n"
    
    header_content += f"""
}};

// Expected output from Python TFLite inference
static const float TEST_VECTOR_EXPECTED_OUTPUT[MODEL_OUTPUT_SIZE] = {{
    {expected_output[0]:.6f}f,  // other_sounds
    {expected_output[1]:.6f}f   // lg_melody
}};

// Test vector metadata
#define TEST_VECTOR_SHAPE_FREQ {spectrogram.shape[0]}
#define TEST_VECTOR_SHAPE_TIME {spectrogram.shape[1]}
#define TEST_VECTOR_SIZE {len(flattened_spec)}

#endif // TEST_VECTOR_H
"""
    
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"Generated C header: {header_path}")

def main():
    print("=== Generating Test Vector from LG Melody Sample ===")
    
    # Process the LG melody audio file
    audio_path = "data/lg_melody/lg.wav"
    print(f"Processing: {audio_path}")
    
    spectrogram = process_audio_file(audio_path, CONFIG)
    if spectrogram is None:
        print("Failed to process audio file!")
        return
    
    print(f"Generated spectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram range: [{np.min(spectrogram):.2f}, {np.max(spectrogram):.2f}] dB")
    
    # Run TFLite inference to get expected output
    model_path = "lg_sound_model.tflite"
    print(f"Running inference with: {model_path}")
    
    try:
        expected_output = run_tflite_inference(model_path, spectrogram)
        print(f"Expected output: {expected_output}")
        print(f"LG melody probability: {expected_output[1]:.6f}")
    except Exception as e:
        print(f"Error running TFLite inference: {e}")
        return
    
    # Generate C header file
    header_path = "esp32_melody_detector/main/test_vector.h"
    generate_c_header(spectrogram, expected_output, header_path)
    
    print("\n=== Test Vector Generation Complete ===")
    print(f"Input shape: {spectrogram.shape}")
    print(f"Input size: {spectrogram.size}")
    print(f"Expected output: [{expected_output[0]:.6f}, {expected_output[1]:.6f}]")
    print(f"Header file: {header_path}")

if __name__ == "__main__":
    main()
