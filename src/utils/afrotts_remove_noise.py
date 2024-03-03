import os
from pathlib import Path
import pandas as pd

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
# !pip install -U denoiser

def clean_audio(
    model,
    data,
    src_dir,
    dest_dir,
    device,
):

    # assuming audio files have been resampled to 16k
    for _, item in data.iterrows():
        src = os.path.join(dir_path, "..",
                           item.audio_paths[1:])
        
        dst = os.path.join(dest_dir, item.audio_paths[1:])
        
        if os.path.exists(dst): continue
        
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        try:
            # You can pass a NumPy array:
            wav, sr = torchaudio.load(src)
            
            wav = convert_audio(wav.to(device), sr, model.sample_rate, model.chin)
            with torch.no_grad():
                denoised = model(wav[None])[0]
                
            torchaudio.save(dst, denoised.data.cpu(), 16000, encoding="PCM_S", bits_per_sample=16)

        except Exception as e:
            print(f"{src} cannot be processed")
            print(e)
            continue

    print(f"Process finished")


def main(data, dir_path):
    """Download audio files, and converts to raw"""

    # model = BaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")

    model = pretrained.dns64()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    clean_audio(
        model,
        data=data,
        src_dir=os.path.join(dir_path, ".."),
        dest_dir=os.path.join(dir_path, "afrispeech_16k_denoised"),
        device=device,
        )


if __name__ == "__main__":
    
    # path to AfriSpeech dir
    dir_path = os.getcwd()
    
    train = pd.read_csv(os.path.join(dir_path, "data/intron-tts-train-public-28565.csv"))
    dev = pd.read_csv(os.path.join(dir_path, "data/intron-tts-dev-public-3330.csv"))
    test = pd.read_csv(os.path.join(dir_path, "data/intron-tts-test-public-4161.csv"))

    data = pd.concat([train, dev, test])
    
    # can limit the max duration to 50 secs for faster computation
    data = data[data.duration <= 50.0].copy()
    
    main(data, dir_path)


# files not found
# /AfriSpeech-TTS/test/965756d8-72df-4570-9f9a-136c6aee6eaf/274d814a07965f4cd8e9655625a4e43e_pJ2UcROg.wav
# /AfriSpeech-TTS/test/f29bb76f-49bd-4b29-9164-d0cd2de12071/2f38ad4d51cfe046750c90bb4ceb7e24_iRR8COgR.wav
# /AfriSpeech-TTS/train/9b1a7865-4b74-486e-8223-9117e2ea592a/9b1f7d24a96824967ed3a1ae5d3c44cd_OWVB6wln.wav

