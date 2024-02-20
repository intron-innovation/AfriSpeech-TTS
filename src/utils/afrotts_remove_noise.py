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
    
    # can limit the max duration to 30 secs for faster computation
    data = data[data.duration <= 200.0].copy()
    
    main(data, dir_path)


# files not found
# /AfriSpeech-TTS/test/965756d8-72df-4570-9f9a-136c6aee6eaf/274d814a07965f4cd8e9655625a4e43e_pJ2UcROg.wav
# /AfriSpeech-TTS/test/f29bb76f-49bd-4b29-9164-d0cd2de12071/2f38ad4d51cfe046750c90bb4ceb7e24_iRR8COgR.wav
# /AfriSpeech-TTS/train/9b1a7865-4b74-486e-8223-9117e2ea592a/9b1f7d24a96824967ed3a1ae5d3c44cd_OWVB6wln.wav

# files too long to be denoised > 100 secs
# /AfriSpeech-TTS/train/233d7b8b-ff36-4f2f-9078-1105779ab6f6/23ccae8f393875b815cc0641664c49be_qVjCaApc.wav
# /AfriSpeech-TTS/train/62b7b1cb-f2ab-4bb6-b8fc-a732ffc01bd0/36ca300614289fef5202f74957278042_O6FbLcv4.wav
# /AfriSpeech-TTS/train/eb4cfb55-74b8-4818-98c5-642bacba29d0/4e7d1a707b73e1ed6395ac9b2517102b_2CccHTjF.wav
# /AfriSpeech-TTS/train/03652986-c81c-4c34-b309-75e53d3169e7/7da692ca4cdc5e666f89cf83fffb85f5_i9bjhBtS.wav
# /AfriSpeech-TTS/train/18352cd8-d4c5-453c-9d28-fd90a1ae163b/5dc07a7cec5e18843583852d30e3e0ca_JCIE8TKi.wav
# /AfriSpeech-TTS/dev/383f7a6c-abc0-4e0c-b871-c197cebc08da/cfcd09a25171d54afe8460d343f384f3_F1bcQe8w.wav
# /AfriSpeech-TTS/dev/9080d8cf-3280-455a-a6cc-23f9645580e2/1d5b191c7ff72989ad34c7fdddf1bf5d_jZfOM8GZ.wav
# /AfriSpeech-TTS/test/77228093-69b4-4c81-bc1f-7537c46b02ce/75bc507c315235f67d4245420199430b_WhBbNTVE.wav
