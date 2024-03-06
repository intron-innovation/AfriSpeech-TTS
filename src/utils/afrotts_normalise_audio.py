

import multiprocessing
import os
import struct
import time

import librosa
import numpy as np
import pandas as pd
import soundfile
# import webrtcvad

from glob import glob

# pip3 install ffmpeg-normalize

## Audio
sampling_rate = 16000

def preprocess_wav(data, dst_path, cpu_proc, target_sr=16000):
    # pip3 install ffmpeg-normalize
    for i, item in data.iterrows():
        src = os.path.join(dir_path, "afrispeech_16k_cleaned",
                           item.audio_paths[1:])

        
        dst = os.path.join(dst_path, item.audio_paths[1:])
        
        if os.path.exists(dst): continue
        
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # normalise volume
        if os.path.exists(src):
            try:
                cmd = f"ffmpeg-normalize {src} -o {dst} -ar 16000 --target-level -27 -nt rms"
                os.system(cmd)
            except Exception:
                print(f"{src} cannot be processed")
                continue

    print(f"Process {cpu_proc} finished")


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

    starttime = time.time()
    
        
    # path to AfriSpeech dir
    dir_path = os.getcwd()
    
    train = pd.read_csv(os.path.join(dir_path, "data/intron-tts-train-public-28565.csv"))
    dev = pd.read_csv(os.path.join(dir_path, "data/intron-tts-dev-public-3330.csv"))
    test = pd.read_csv(os.path.join(dir_path, "data/intron-tts-test-public-4161.csv"))

    data = pd.concat([train, dev, test])
    data = data.sample(frac=1)
    
    # limit the max duration to 50 secs for faster computation
    data = data[data.duration <= 50.0].copy()

    dst_path = os.path.join(dir_path, "afrispeech_16k_norm")
    main(data, dst_path)

    print("That took {} seconds".format(time.time() - starttime))
