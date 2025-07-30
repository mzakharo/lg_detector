import librosa
import numpy as np

# --- Configuration (MUST MATCH YOUR TRAINING SCRIPT AND C++ CODE) ---
SAMPLE_RATE = 16000
N_FFT = 1024
N_MELS = 64
FMIN = 0
FMAX = SAMPLE_RATE / 2

def generate_mel_filterbank_header(filename="mel_filters.h"):
    """Generates a C header file containing the Mel filterbank matrix."""
    mel_filters = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        norm=None, # Use 'slaney' norm if your training used it
    ).T # Transpose to get shape [n_fft // 2 + 1, n_mels]

    with open(filename, 'w') as f:
        f.write("#pragma once\n\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"const int MEL_FILTER_BANK_N_MELS = {N_MELS};\n")
        f.write(f"const int MEL_FILTER_BANK_N_FFT_BINS = {N_FFT // 2 + 1};\n\n")
        f.write("const float mel_filter_bank[MEL_FILTER_BANK_N_FFT_BINS][MEL_FILTER_BANK_N_MELS] = {\n")

        for i in range(mel_filters.shape[0]):
            f.write("    {")
            for j in range(mel_filters.shape[1]):
                f.write(f"{mel_filters[i, j]:.8f}f, ")
            f.write("},\n")
        f.write("};\n")
    print(f"Mel filterbank saved to {filename}")

if __name__ == "__main__":
    generate_mel_filterbank_header("main/mel_filters.h")
