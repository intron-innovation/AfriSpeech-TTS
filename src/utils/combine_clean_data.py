
import os
import numpy as np
import shutil
import pandas as pd

import librosa
import soundfile

if __name__ == "__main__":
    # filter files with low mos scores
    dir_path = os.getcwd()

    train = pd.read_csv(os.path.join(dir_path, "data/intron-tts-train-public-28565.csv"))
    dev = pd.read_csv(os.path.join(dir_path, "data/intron-tts-dev-public-3330.csv"))
    test = pd.read_csv(os.path.join(dir_path, "data/intron-tts-test-public-4161.csv"))

    data = pd.concat([train, dev, test]) # train[20271:] 
    

    from wvmos import get_wvmos
    wvmos_model = get_wvmos(cuda=True)

    # can limit the max duration to 50 secs for faster computation
    data = data[data.duration <= 50.0].copy()
    modes = ["dn", "0", "1", "2"]
    with open("/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sogun/AfriSpeech-TTS/src/utils/mos_files_d1123_3.txt", "a+") as f:
        for i, item in data.iterrows():
            mode_temp, mos_scores = [], []
            
            dst = os.path.join("/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sogun/AfriSpeech-TTS/afrispeech_16k_cleaned",
                item.audio_paths[1:])
            if os.path.exists(dst): continue
            
            for mode in modes:
                if mode == "dn":
                    src = os.path.join("/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sogun/AfriSpeech-TTS/afrispeech_16k_denoised",
                                item.audio_paths[1:])
                else:
                    src = os.path.join("/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sogun/AfriSpeech-TTS/afrispeech_voicefix",
                                mode, item.audio_paths[1:])
                if not os.path.exists(src): continue
                mos_score = wvmos_model.calculate_one(src)
                mos_scores.append(mos_score)
                mode_temp.append(mode)
            
            if len(mos_scores) > 0:
                best_mos_idx = np.argmax(mos_scores)

                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if modes[best_mos_idx] == "dn":
                    src = os.path.join("/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sogun/AfriSpeech-TTS/afrispeech_16k_denoised",
                                item.audio_paths[1:])

                    shutil.copy2(src, dst)
                else:
                    
                    src = os.path.join("/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sogun/AfriSpeech-TTS/afrispeech_voicefix",
                                modes[best_mos_idx], item.audio_paths[1:])
                    audio, sr = librosa.load(src, sr=16000)
                    soundfile.write(dst, audio.squeeze(), sr)
            else:
                print(f"file missing: {item.audio_ids}")
                mos_scores, mode_temp = [], []
                continue

            mos_scores = [str(x) for x in mos_scores]
            str_mos = "|".join(mos_scores)
            print(f"{item.audio_ids}|{str_mos}", file=f)
            
            mos_scores, mode_temp = [], []
            
    print("finished")