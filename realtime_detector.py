import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

#import tflite_runtime.interpreter as tflite
import queue

# --- CONFIGURATION (MUST MATCH YOUR TRAINING SCRIPT) ---
# This dictionary holds all the crucial parameters.
# Ensure these are identical to the ones used to create the model.
CONFIG = {
    "model_path": "lg_sound_model.tflite",
    "class_names": ["other_sounds", "lg_melody"], # IMPORTANT: Order must match training
    "sample_rate": 16000,
    "window_duration": 1.5,  # seconds (length of one spectrogram)
    "hop_duration": 0.5,     # seconds (how much to slide the window)
    "n_mels": 64,
    "n_fft": 1024,
    "max_spectrogram_width": 48
}

# --- CONFIDENCE ACCUMULATOR PARAMETERS ---
CONFIDENCE_PARAMS = {
    "prediction_threshold": 0.80, # Raw model output to be considered a 'positive' frame
    "confidence_threshold": 0.95, # Accumulated score to trigger the final alert
    "increment_amount": 0.15,
    "decay_rate": 0.05,
    "cooldown_seconds": 10 # Seconds to wait before detecting again
}

# Global variables for the audio stream and processing
audio_queue = queue.Queue()
confidence_score = 0.0
is_in_cooldown = False
last_trigger_time = 0

def load_tflite_model(model_path):
    """Loads the TFLite model and allocates tensors."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def audio_callback(indata, frames, time, status):
    """This function is called by the sounddevice stream for each new audio chunk."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def preprocess_window(audio_window, config):
    """Converts an audio window to a spectrogram suitable for the model."""
    # Generate Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=audio_window, 
        sr=config["sample_rate"], 
        n_mels=config["n_mels"], 
        n_fft=config["n_fft"]
    )
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Standardize spectrogram width
    if log_spectrogram.shape[1] > config["max_spectrogram_width"]:
        log_spectrogram = log_spectrogram[:, :config["max_spectrogram_width"]]
    else:
        pad_width = config["max_spectrogram_width"] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        
    # Reshape for the model (add batch and channel dimensions)
    # and ensure it's the correct data type (float32).
    return log_spectrogram.astype(np.float32)[np.newaxis, ..., np.newaxis]


def main():
    global confidence_score, is_in_cooldown, last_trigger_time

    # --- SETUP ---
    print("Loading TFLite model...")
    interpreter = load_tflite_model(CONFIG["model_path"])
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully.")
    
    # Calculate required buffer sizes based on config
    window_samples = int(CONFIG["window_duration"] * CONFIG["sample_rate"])
    hop_samples = int(CONFIG["hop_duration"] * CONFIG["sample_rate"])
    
    # This buffer will hold the continuous audio stream
    audio_buffer = np.array([], dtype=np.float32)

    # --- START REAL-TIME PROCESSING ---
    try:
        print("\nStarting microphone stream... Press Ctrl+C to stop.")
        stream = sd.InputStream(
            callback=audio_callback, 
            channels=1, 
            samplerate=CONFIG["sample_rate"],
            dtype='float32'
        )
        with stream:
            while True:
                # Get audio data from the queue (put there by the callback)
                new_audio = audio_queue.get()
                audio_buffer = np.append(audio_buffer, new_audio)

                # Process the buffer only if we have enough data for a full window
                while len(audio_buffer) >= window_samples:
                    # 1. Handle Cooldown
                    if is_in_cooldown and (librosa.time_to_samples(CONFIDENCE_PARAMS["cooldown_seconds"], sr=CONFIG["sample_rate"]) < (len(audio_buffer) - last_trigger_time)):
                         is_in_cooldown = False
                         print("Cooldown finished. Resuming detection.")

                    if not is_in_cooldown:
                        # 2. Preprocess a window of audio
                        window = audio_buffer[:window_samples]
                        spectrogram = preprocess_window(window, CONFIG)

                        # 3. Run Inference
                        interpreter.set_tensor(input_details[0]['index'], spectrogram)
                        interpreter.invoke()
                        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
                        
                        # Assuming the 'lg_melody' class is the second one
                        melody_prob = prediction[1]

                        # 4. Update Confidence Score
                        if melody_prob > CONFIDENCE_PARAMS["prediction_threshold"]:
                            confidence_score += CONFIDENCE_PARAMS["increment_amount"]
                            confidence_score = min(confidence_score, 1.0)
                        else:
                            confidence_score -= CONFIDENCE_PARAMS["decay_rate"]
                            confidence_score = max(confidence_score, 0.0)
                        
                        # Print status update
                        print(f"\rMelody Probability: {melody_prob:.2f} | Confidence: [{confidence_score:.2f}] {'#' * int(confidence_score * 20):<20}", end="")

                        # 5. Check for Trigger
                        if confidence_score >= CONFIDENCE_PARAMS["confidence_threshold"]:
                            print("\n\n*** MELODY DETECTED! ***\n")
                            is_in_cooldown = True
                            last_trigger_time = len(audio_buffer) # Mark time by buffer position
                            confidence_score = 0.0 # Reset immediately
                            # You could add other actions here (e.g., send a notification)
                    
                    # 6. Slide the buffer forward by the hop amount
                    audio_buffer = audio_buffer[hop_samples:]

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()