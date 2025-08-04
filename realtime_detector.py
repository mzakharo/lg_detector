import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import paho.mqtt.client as mqtt
import struct
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import threading
import time


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
spectrogram_queue = queue.Queue()
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


def on_connect(client, userdata, flags, rc):
    """The callback for when the client receives a CONNACK response from the server."""
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(CONFIG["mqtt_topic"])
        print(f"Subscribed to topic: {CONFIG['mqtt_topic']}")
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    """The callback for when a PUBLISH message is received from the server."""
    try:
        # ESP32 sends as little-endian float32
        expected_bytes = CONFIG["n_mels"] * CONFIG["max_spectrogram_width"] * 4
        if len(msg.payload) != expected_bytes:
            print(f"Warning: Received {len(msg.payload)} bytes, expected {expected_bytes}")
            return

        float_data = struct.unpack(f'<{CONFIG["n_mels"] * CONFIG["max_spectrogram_width"]}f', msg.payload)
        # The C++ code now sends the spectrogram in the correct [mels, time] orientation.
        # We just need to reshape it. The old transpose is no longer needed.
        spectrogram = np.array(float_data).reshape(CONFIG["n_mels"], CONFIG["max_spectrogram_width"])
        
        # The model expects the spectrogram in a specific shape and type
        # Reshape for the model (add batch and channel dimensions)
        model_spectrogram = spectrogram.astype(np.float32)[np.newaxis, ..., np.newaxis]
        
        spectrogram_queue.put((model_spectrogram, spectrogram))

    except Exception as e:
        print(f"Error processing MQTT message: {e}")


class SpectrogramVisualizer:
    def __init__(self, n_mels, spectrogram_width):
        self.n_mels = n_mels
        self.spectrogram_width = spectrogram_width
        self.latest_spectrogram = np.zeros((self.n_mels, self.spectrogram_width))
        self.data_lock = threading.Lock()

        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.fig.suptitle('Real-time Spectrogram', fontsize=14)
        
        colors = ['#000033', '#000055', '#0000ff', '#0055ff', '#00ffff', '#55ff00', '#ffff00', '#ff5500', '#ff0000', '#ffffff']
        cmap = LinearSegmentedColormap.from_list('spectrogram', colors, N=256)
        
        self.im = self.ax.imshow(self.latest_spectrogram, aspect='auto', origin='lower', cmap=cmap, vmin=-80, vmax=0, interpolation='nearest')
        self.ax.set_xlabel('Time Frames')
        self.ax.set_ylabel('Mel Frequency Bins')
        self.ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(self.im, ax=self.ax)
        cbar.set_label('Power (dB)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def update_spectrogram(self, new_spectrogram):
        with self.data_lock:
            self.latest_spectrogram = new_spectrogram

    def update_plot(self, frame):
        with self.data_lock:
            self.im.set_array(self.latest_spectrogram)
        return [self.im]

    def draw(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def start(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        #plt.ion()
        plt.show(block=False)

    def stop(self):
        plt.close(self.fig)

def main():
    global confidence_score, is_in_cooldown, last_trigger_time

    parser = argparse.ArgumentParser(description='Real-time LG sound detector.')
    parser.add_argument('--input', type=str, default='mic', choices=['mic', 'mqtt'],
                        help="Input source: 'mic' for microphone or 'mqtt' for MQTT.")
    parser.add_argument('--broker', type=str, default='nas.local', help='MQTT broker address.')
    parser.add_argument('--topic', type=str, default='lg-detector/spectrogram', help='MQTT topic for spectrograms.')
    parser.add_argument('--visualize', action='store_true', help='Enable real-time spectrogram visualization.')
    args = parser.parse_args()

    CONFIG["input_source"] = args.input
    CONFIG["mqtt_broker"] = args.broker
    CONFIG["mqtt_topic"] = args.topic


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
    if CONFIG["input_source"] == 'mqtt':
        print(f"\nConnecting to MQTT broker at {CONFIG['mqtt_broker']}...")
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(CONFIG['mqtt_broker'], 1883, 60)
        client.loop_start()
    else:
        print("\nStarting microphone stream... Press Ctrl+C to stop.")
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=CONFIG["sample_rate"],
            dtype='float32'
        )
        stream.start()

    visualizer = None
    if args.visualize:
        visualizer = SpectrogramVisualizer(CONFIG["n_mels"], CONFIG["max_spectrogram_width"])
        visualizer.start()

    try:
        while True:
            model_spectrogram = None
            display_spectrogram = None

            if CONFIG["input_source"] == 'mic':
                new_audio = audio_queue.get()
                audio_buffer = np.append(audio_buffer, new_audio)

                while len(audio_buffer) >= window_samples:
                    window = audio_buffer[:window_samples]
                    
                    # Create spectrogram for model
                    model_spectrogram = preprocess_window(window, CONFIG)
                    
                    # Create spectrogram for display (without batch/channel dims)
                    if visualizer:
                        display_spectrogram = np.squeeze(model_spectrogram)

                    audio_buffer = audio_buffer[hop_samples:]
                    break # Process one window at a time

            else: # mqtt
                model_spectrogram, display_spectrogram = spectrogram_queue.get()

            if model_spectrogram is not None:
                if visualizer and display_spectrogram is not None:
                    visualizer.update_spectrogram(display_spectrogram)
                    visualizer.draw()

                # 3. Run Inference
                interpreter.set_tensor(input_details[0]['index'], model_spectrogram)
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
                    last_trigger_time = len(audio_buffer) # Mark time by buffer position
                    confidence_score = 0.0 # Reset immediately
                    # You could add other actions here (e.g., send a notification)
            

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if visualizer:
            visualizer.stop()
        if CONFIG["input_source"] == 'mqtt' and 'client' in locals():
            client.loop_stop()
        elif 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
