import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class AudioConfig:
    sr = 16000
    duration = 3.0  # secs
    min_audio_len = sr * duration


def pad_zeros(x, size, sr):
    if len(x) >= size:  # long enough
        return x
    else:  # pad blank
        return np.pad(x, (0, max(0, sr - len(x))), "constant")


def load_audio_file(file_path):
    try:
        data, sr = librosa.core.load(file_path, sr=AudioConfig.sr)
        if sr != AudioConfig.sr:
            data = librosa.resample(data, sr, AudioConfig.sr)
        if len(data) < sr:
            data = pad_zeros(data, AudioConfig.sr, AudioConfig.sr)
    except Exception as e:
        print(f"{file_path} not found {str(e)}")
        data = np.random.rand(AudioConfig.sr * 3).astype(np.float32)
    return data
