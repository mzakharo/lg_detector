#!/usr/bin/env python3
"""
Real-time MQTT Spectrogram Viewer for ESP32 LG Detector
Connects to MQTT broker and visualizes incoming spectrogram data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import paho.mqtt.client as mqtt
import struct
import threading
import time
from collections import deque
import argparse

class MQTTSpectrogramViewer:
    def __init__(self, broker_host="nas.local", broker_port=1883, topic="lg-detector/spectrogram"):
        # MQTT Configuration
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        
        # Spectrogram parameters (must match ESP32 configuration)
        self.N_MELS = 64
        self.SPECTROGRAM_WIDTH = 48
        self.EXPECTED_BYTES = self.N_MELS * self.SPECTROGRAM_WIDTH * 4  # 4 bytes per float
        
        # Data storage
        self.spectrogram_history = deque(maxlen=200)  # Store last 200 spectrograms
        self.latest_spectrogram = None
        self.data_lock = threading.Lock()
        
        # Statistics
        self.message_count = 0
        self.last_message_time = 0
        self.message_rate = 0.0
        
        # MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Matplotlib setup
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('ESP32 LG Detector - Real-time Spectrogram Viewer', fontsize=14)
        
        # Create custom colormap (similar to librosa's default)
        colors = ['#000033', '#000055', '#0000ff', '#0055ff', '#00ffff', '#55ff00', '#ffff00', '#ff5500', '#ff0000', '#ffffff']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('spectrogram', colors, N=n_bins)
        
        # Initialize plots
        self.im1 = self.ax1.imshow(np.zeros((self.N_MELS, self.SPECTROGRAM_WIDTH)), 
                                   aspect='auto', origin='lower', cmap=cmap, 
                                   vmin=-80, vmax=0, interpolation='nearest')
        self.ax1.set_title('Current Spectrogram (48 time frames × 40 mel frequencies)')
        self.ax1.set_xlabel('Time Frames')
        self.ax1.set_ylabel('Mel Frequency Bins')
        self.ax1.grid(True, alpha=0.3)
        
        # Add colorbar for current spectrogram
        cbar1 = plt.colorbar(self.im1, ax=self.ax1)
        cbar1.set_label('Power (dB)')
        
        # History plot (waterfall display)
        self.im2 = self.ax2.imshow(np.zeros((100, self.SPECTROGRAM_WIDTH)), 
                                   aspect='auto', origin='upper', cmap=cmap,
                                   vmin=-80, vmax=0, interpolation='nearest')
        self.ax2.set_title('Spectrogram History (Time flows downward)')
        self.ax2.set_xlabel('Time Frames')
        self.ax2.set_ylabel('History (newest at top)')
        self.ax2.grid(True, alpha=0.3)
        
        # Add colorbar for history
        cbar2 = plt.colorbar(self.im2, ax=self.ax2)
        cbar2.set_label('Average Power (dB)')
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, 'Status: Connecting...', fontsize=10)
        
        plt.tight_layout()
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            client.subscribe(self.topic)
            print(f"Subscribed to topic: {self.topic}")
        else:
            print(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        print(f"Disconnected from MQTT broker. Return code: {rc}")
    
    def on_message(self, client, userdata, msg):
        try:
            # Check message size
            if len(msg.payload) != self.EXPECTED_BYTES:
                print(f"Warning: Received {len(msg.payload)} bytes, expected {self.EXPECTED_BYTES}")
                return
            
            # Convert binary data to numpy array
            # ESP32 sends as little-endian float32
            float_data = struct.unpack(f'<{self.N_MELS * self.SPECTROGRAM_WIDTH}f', msg.payload)
            #spectrogram = np.array(float_data).reshape(self.SPECTROGRAM_WIDTH, self.N_MELS)
            #spectrogram = spectrogram.T
            spectrogram = np.array(float_data).reshape(self.N_MELS, self.SPECTROGRAM_WIDTH)
            
            # Update data with thread safety
            with self.data_lock:
                self.latest_spectrogram = spectrogram
                self.spectrogram_history.append(np.mean(spectrogram, axis=0))  # Average across mel bins for history
                
                # Update statistics
                current_time = time.time()
                if self.last_message_time > 0:
                    time_diff = current_time - self.last_message_time
                    self.message_rate = 0.9 * self.message_rate + 0.1 * (1.0 / time_diff)  # Exponential smoothing
                self.last_message_time = current_time
                self.message_count += 1
            
            # Print statistics every 10 messages
            if self.message_count % 10 == 0:
                min_val = np.min(spectrogram)
                max_val = np.max(spectrogram)
                mean_val = np.mean(spectrogram)
                print(f"Message {self.message_count}: Rate={self.message_rate:.1f} Hz, "
                      f"Range=[{min_val:.1f}, {max_val:.1f}] dB, Mean={mean_val:.1f} dB")
                
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def update_plot(self, frame):
        with self.data_lock:
            if self.latest_spectrogram is not None:
                # Update current spectrogram
                self.im1.set_array(self.latest_spectrogram)
                
                # Update history plot
                if len(self.spectrogram_history) > 0:
                    history_data = np.array(list(self.spectrogram_history))
                    # Pad or truncate to fit display
                    display_height = 100
                    if len(history_data) < display_height:
                        # Pad with zeros at the bottom
                        padded_data = np.zeros((display_height, self.SPECTROGRAM_WIDTH))
                        padded_data[:len(history_data)] = history_data
                        history_data = padded_data
                    else:
                        # Take the most recent data
                        history_data = history_data[-display_height:]
                    
                    self.im2.set_array(history_data)
                
                # Update status
                status = f"Messages: {self.message_count} | Rate: {self.message_rate:.1f} Hz | "
                if self.latest_spectrogram is not None:
                    status += f"Range: [{np.min(self.latest_spectrogram):.1f}, {np.max(self.latest_spectrogram):.1f}] dB"
                else:
                    status += "No data"
                self.status_text.set_text(status)
        
        return [self.im1, self.im2, self.status_text]
    
    def start(self):
        # Connect to MQTT broker
        try:
            print(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}...")
            self.mqtt_client.connect(self.broker_host, self.broker_port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            return
        
        # Start animation
        print("Starting real-time visualization...")
        print("Close the plot window to exit.")
        
        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

def main():
    parser = argparse.ArgumentParser(description='Real-time MQTT Spectrogram Viewer for ESP32 LG Detector')
    parser.add_argument('--broker', default='nas.local', help='MQTT broker hostname (default: nas.local)')
    parser.add_argument('--port', type=int, default=1883, help='MQTT broker port (default: 1883)')
    parser.add_argument('--topic', default='lg-detector/spectrogram', help='MQTT topic (default: lg-detector/spectrogram)')
    
    args = parser.parse_args()
    
    print("ESP32 LG Detector - Real-time Spectrogram Viewer")
    print("=" * 50)
    print(f"MQTT Broker: {args.broker}:{args.port}")
    print(f"MQTT Topic: {args.topic}")
    print(f"Expected data: 40 mel bins × 48 time frames = 7680 bytes per message")
    print()
    
    viewer = MQTTSpectrogramViewer(args.broker, args.port, args.topic)
    viewer.start()

if __name__ == "__main__":
    main()
