"""This code is adapted for common voice preprocessing from https://github.com/resemble-ai/Resemblyzer/blob/199c632495cfd288ca63b790e789616c91d44a01/resemblyzer/audio.py"""

import multiprocessing
import os
import struct
import time

import librosa
import numpy as np
import pandas as pd
import soundfile
import webrtcvad

# from resemblyzer.hparams import *
from scipy.ndimage.morphology import binary_dilation

from glob import glob


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160  # 1600 ms


## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out.
vad_moving_average_width = 24 # 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 4


## Audio volume normalization
audio_norm_target_dBFS = -20


## Model parameters
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3
int16_max = (2 ** 15) - 1


def preprocess_wav(data, dst_path, cpu_proc, target_sr=16000):
    # for (_, file_name, *_) in dataframe.iterrows():
    for i, item in data.iterrows():
        src = os.path.join(dir_path, "afrispeech_16k_norm",
                           item.audio_paths[1:])

        
        dst = os.path.join(dst_path, item.audio_paths[1:])
        
        if os.path.exists(dst): continue
        
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # convert mp3 to wav and trim silence
        try:
            wav, source_sr = librosa.load(str(src), sr=sampling_rate)

            # Apply the preprocessing: shorten long silences
            wav_temp = trim_long_silences(wav)
            
            # cater for case where sound is too low and vad views the entire utterance as silence
            duration = len(wav_temp)/source_sr
            if duration == 0.0:
                print(f"Warning: file {file_name_wav} has a 0 duration")
                wav_temp = wav

            soundfile.write(file=dst, data=wav_temp, samplerate=target_sr, format=None)

        except Exception:
            print(f"File {item.audio_paths} unable to be processed")
            continue

    print(f"Process {cpu_proc} finished")


# Smooth the voice detection with a moving average
def moving_average(array, width):
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    ret = np.cumsum(array_padded, dtype=float)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1 :] / width


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[: len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(
            vad.is_speech(pcm_wave[window_start * 2 : window_end * 2], sample_rate=sampling_rate)
        )
    voice_flags = np.array(voice_flags)

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def main(data, dst_path):
    """Download audio files, and converts to raw"""

    processes = []

    num_cpu = multiprocessing.cpu_count() // 2
    # num_cpu = num_cpu // 2

    print(f"number of cpus available is {num_cpu}")
    total_splits = len(data) // num_cpu

    print(f"{num_cpu} cpu(s) available for parallel conversion to raw")

    for i in range(num_cpu - 1):
        current_split = data[
            int(total_splits * i) : int(total_splits * i + total_splits)
        ]
        proc = multiprocessing.Process(
            target=preprocess_wav,
            args=(
                current_split,
                dst_path,
                i,
            ),
        )
        processes.append(proc)
        proc.start()
    final_split = data[total_splits * (num_cpu - 1) :]

    proc = multiprocessing.Process(
        target=preprocess_wav,
        args=(
            final_split,
            dst_path,
            num_cpu - 1,
        ),
    )
    processes.append(proc)
    proc.start()

    # complete the processes
    for proc in processes:
        proc.join()

    print("Conversion to wav completed")


if __name__ == "__main__":
    ### to be performed after volume normalization !!!
    # pip install resemblyzer
    
    starttime = time.time()
    
        
    # path to AfriSpeech dir
    dir_path = os.getcwd()
    
    train = pd.read_csv(os.path.join(dir_path, "data/intron-tts-train-public-28565.csv"))
    dev = pd.read_csv(os.path.join(dir_path, "data/intron-tts-dev-public-3330.csv"))
    test = pd.read_csv(os.path.join(dir_path, "data/intron-tts-test-public-4161.csv"))

    data = pd.concat([train, dev, test])
    
    # limit the max duration to 30 secs for faster computation
    # data = data[data.duration <= 100.0].copy()

    dst_path = os.path.join(dir_path, "afrispeech_16k_trimmed")
    main(data, dst_path)

    print("That took {} seconds".format(time.time() - starttime))
