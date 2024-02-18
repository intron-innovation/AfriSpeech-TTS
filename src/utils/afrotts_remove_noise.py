import os
from pathlib import Path
import pandas as pd
import soundfile as sf
import torch
from asteroid.models import BaseModel


def clean_audio(
    model,
    data,
    src_dir,
    dest_dir,
):

    # assuming audio files have been resampled to 16k
    for _, item in data.iterrows():
        file_name = f"{item.audio_ids}.wav"
        
        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dest_dir, file_name)

        if os.path.exists(dst): continue
        
        try:
            # You can pass a NumPy array:
            mixture, sr = sf.read(src, dtype="float32", always_2d=True,)
            
            # Soundfile returns the mixture as shape (time, channels), and Asteroid expects (batch, channels, time)
            mixture = mixture.transpose()
            mixture = mixture.reshape(1, mixture.shape[0], mixture.shape[1])

            out_wav = model.separate(mixture)
            out_wav = out_wav.squeeze()

            sf.write(dst, out_wav, sr)

        except Exception as e:
            print(f"{src} cannot be processed")
            print(e)
            continue

    print(f"Process finished")


def main(data, dir_path):
    """Download audio files, and converts to raw"""

    # model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")

    model = BaseModel.from_pretrained("JorisCos/DPTNet_Libri1Mix_enhsingle_16k")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    clean_audio(
        model,
        data=data,
        src_dir=os.path.join(dir_path, "afrispeech_16k"),
        dest_dir=os.path.join(dir_path, "afrispeech_16k_clean"),
        )


if __name__ == "__main__":
    
    # path to AfriSpeech dir
    dir_path = os.getcwd()
    
    train = pd.read_csv(os.path.join(dir_path, "data/intron-tts-train-public-28565.csv"))
    dev = pd.read_csv(os.path.join(dir_path, "data/intron-tts-dev-public-3330.csv"))
    test = pd.read_csv(os.path.join(dir_path, "data/intron-tts-test-public-4161.csv"))

    data = pd.concat([train, dev, test])
    
    # limit the max duration to 30 secs for faster computation
    data = data[data.duration <= 30.0].copy()
    
    main(data, dir_path)


# files not found
# /AfriSpeech-TTS/afrispeech_16k/24bfceabfa102cc1c1926d2049f56bbf.wav cannot be processed
# /AfriSpeech-TTS/afrispeech_16k/790cdbb7907d0112bc737f0b06282dde.wav cannot be processed
# /AfriSpeech-TTS/afrispeech_16k/4bffcdb03a445616d70eb1a859ac52a5.wav cannot be processed
