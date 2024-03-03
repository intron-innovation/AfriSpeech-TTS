import os
from pathlib import Path
import pandas as pd

import torch
from voicefixer import VoiceFixer
# !pip install -U denoiser

def clean_audio(
    model,
    data,
    dir_path,
    dst_path,
    device,
):

    # or voicefixer = VoiceFixer(model='voicefixer/voicefixer')
    # Mode 0: Original Model (suggested by default)
    # Mode 1: Add preprocessing module (remove higher frequency)
    # Mode 2: Train mode (might work sometimes on seriously degraded real speech)
    for i, item in data.iterrows():
        for mode in [0, 1, 2]:
            src = os.path.join(dir_path, "afrispeech_16k_denoised",
                            item.audio_paths[1:])
        
            dst = os.path.join(dst_path, str(mode), item.audio_paths[1:])
            
            if os.path.exists(dst): continue
            
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        
            # normalise volume
            if os.path.exists(src):
                try:
                    model.restore(
                        input=src, # low quality .wav/.flac file
                        output=dst, # save file path
                        cuda=device, # GPU acceleration
                        mode=mode
                    )
                except Exception:
                    print(f"{src} cannot be processed")
                    continue

    print(f"Process finished")


def main(data, dir_path):
    """Download audio files, and converts to raw"""

    # !pip install voicefixer

    vf = VoiceFixer()
    device = True if torch.cuda.is_available() else False
    # model = model.to(device)

    clean_audio(
        model=vf,
        data=data,
        dir_path=dir_path,
        dst_path=os.path.join(dir_path, "afrispeech_voicefix"),
        device=device,
        )


if __name__ == "__main__":
    
    # path to AfriSpeech dir
    dir_path = os.getcwd()
    
    train = pd.read_csv(os.path.join(dir_path, "data/intron-tts-train-public-28565.csv"))
    dev = pd.read_csv(os.path.join(dir_path, "data/intron-tts-dev-public-3330.csv"))
    test = pd.read_csv(os.path.join(dir_path, "data/intron-tts-test-public-4161.csv"))

    data = pd.concat([train, dev, test])
    data = data.sample(frac=1).reset_index(drop=True)
    
    # can limit the max duration to 50 secs for faster computation
    data = data[data.duration <= 50.0].copy()
    
    main(data, dir_path)


# files not found
# /AfriSpeech-TTS/test/965756d8-72df-4570-9f9a-136c6aee6eaf/274d814a07965f4cd8e9655625a4e43e_pJ2UcROg.wav
# /AfriSpeech-TTS/test/f29bb76f-49bd-4b29-9164-d0cd2de12071/2f38ad4d51cfe046750c90bb4ceb7e24_iRR8COgR.wav
# /AfriSpeech-TTS/train/9b1a7865-4b74-486e-8223-9117e2ea592a/9b1f7d24a96824967ed3a1ae5d3c44cd_OWVB6wln.wav

