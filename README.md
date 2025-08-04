# LG Appliance Sound Detection System

A machine learning-powered system that detects the completion melody of LG washing machines and other appliances using audio analysis. The system consists of a Python-based training and real-time detection pipeline, plus an ESP32-based embedded detector for standalone operation.

## Overview

This project uses deep learning to identify the distinctive melody that LG appliances play when they complete their cycles. The system can run in two modes:
- **Python Real-time Detector**: Uses a computer's microphone or MQTT data stream
- **ESP32 Embedded Detector**: Standalone device with PDM microphone and WiFi connectivity
- **Tested** with Seeed Studio XIAO ESP32S3 Sense board	
- **Home Assistant** integration over MQTT

## Features

- **Real-time Audio Processing**: Converts audio to mel spectrograms for ML inference
- **Confidence-based Detection**: Uses accumulative scoring to reduce false positives
- **Multiple Input Sources**: Microphone input or MQTT streaming
- **ESP32 Implementation**: Low-power embedded solution with WiFi/MQTT
- **Visualization**: Real-time spectrogram display for debugging
- **Model Training Pipeline**: Complete workflow from data collection to deployment

## System Architecture

```
Audio Input → Spectrogram Generation → ML Model → Confidence Accumulator → Detection Alert
```

### Key Components

1. **Audio Processing**: 16kHz sampling, 1.5s windows, 64 mel bands
2. **CNN Model**: Lightweight 3-layer CNN optimized for embedded deployment
3. **Confidence System**: Accumulative scoring with cooldown periods
4. **Communication**: MQTT for remote monitoring and data streaming

## Quick Start

### Python Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run real-time detection with microphone
python realtime_detector.py --input mic --visualize

# Run with MQTT input (for ESP32 integration)
python realtime_detector.py --input mqtt --broker nas.local --topic lg-detector/spectrogram
```

### ESP32 Setup

1. **Hardware Requirements**:
   - ESP32-S3 or similar with PSRAM
   - PDM microphone (CLK: GPIO42, DATA: GPIO41)
   - WiFi connectivity

2. **Build and Flash**:
    - source esp-idf (v5.4 tested)
   ```bash
   cd esp-lg-detector
   idf.py set-target esp32s3
   idf.py build
   idf.py flash monitor
   ```

3. **Configuration**:
   - Update WiFi credentials in `main.cpp`
   - Set MQTT broker address
   - Adjust detection thresholds if needed

4. **Home Assistant**:
    - Ensure MQTT integration is configured
    - in configuration.yaml
    ```yaml
    mqtt:
            sensor:
                - name: LG_conf 
                state_topic: lg-detector/detection
                unit_of_measurement: "%"
                unique_id: lg_conf 
                value_template: "{{ value_json.confidence_score }}"
                force_update: true
                expire_after: 60
    ```
    - Perform a quick restart of Home Assistant to reload MQTT entities
    - Add Helper Entity. Threshold. Set Trigger threshold % (60). 

## Model Training

The system uses a CNN trained on mel spectrograms:

### Data Structure
```
data/
├── lg_melody/          # Target LG appliance sounds
│   ├── lg.wav
│   ├── lg_washer.wav
│   └── lg_washer_machine.wav
└── other_sounds/       # Background/negative samples
    └── ... (150+ samples)
```

### Training Process

1. **Data Preprocessing**: Audio files → Mel spectrograms (64×48)
2. **Model Architecture**: 3-layer CNN with dropout
3. **Training**: 30 epochs with early stopping
4. **Conversion**: Keras → TensorFlow Lite for deployment

Run the training notebook:
```bash
jupyter notebook notebook.ipynb
```

### ESP32 Operation
The ESP32 automatically:
1. Connects to configured WiFi network
2. Starts PDM microphone capture
3. Processes audio in real-time
4. Publishes detection results via MQTT
5. Logs detection events to serial console

### MQTT Topics
- `lg-detector/spectrogram`: Raw spectrogram data (ESP32 → Python)
- `lg-detector/detection`: Detection results JSON

## Model Performance

- **Input**: 64×48 mel spectrogram (1.5s audio window)
- **Output**: Binary classification [other_sounds, lg_melody]
- **Model Size**: ~50KB TensorFlow Lite
- **Memory Usage**: ~128KB tensor arena

## Hardware Requirements

### Python Version
- Python 3.7+
- Audio input device (microphone)

### ESP32 Version
- ESP32-S3 or similar with PSRAM
- PDM microphone module
- WiFi connectivity

## Troubleshooting

### Common Issues

### Debug Mode
Enable debug output in ESP32:
```cpp
#define DEBUG_MODE 1  // Sine wave test mode

#define SEPCTOGRAM_DEBUG  // Output spectogram over MQTT
```

### Visualization
Use the `--visualize` flag to see real-time spectrograms and debug audio processing.

## Contributing

## Technical Details

### Signal Processing Pipeline
1. **Audio Capture**: 16kHz mono PCM
2. **Windowing**: 1.5s Hann windows
3. **FFT**: 1024-point real FFT
4. **Mel Filtering**: 64-band mel filterbank (0-8kHz)
5. **Log Conversion**: Power to dB scale
6. **Normalization**: Reference-based scaling

### Machine Learning Pipeline
1. **Feature Extraction**: Mel spectrograms as 2D images
2. **Model Architecture**: CNN with 3 conv layers + dense classifier
3. **Training**: Supervised learning on labeled audio samples
4. **Deployment**: TensorFlow Lite for cross-platform inference
5. **Post-processing**: Confidence accumulation with temporal smoothing

