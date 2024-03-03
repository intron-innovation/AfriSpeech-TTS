
import os
import pandas as pd

if __name__ == "__main__":
    # filter files with low mos scores
    dir_path = os.getcwd()

    train = pd.read_csv(os.path.join(dir_path, "data/intron-tts-train-public-28565.csv"))
    dev = pd.read_csv(os.path.join(dir_path, "data/intron-tts-dev-public-3330.csv"))
    test = pd.read_csv(os.path.join(dir_path, "data/intron-tts-test-public-4161.csv"))

    data = pd.concat([train, dev, test])

    from wvmos import get_wvmos
    wvmos_model = get_wvmos(cuda=True)

    # can limit the max duration to 50 secs for faster computation
    data = data[data.duration <= 50.0].copy()

    with open("/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sogun/AfriSpeech-TTS/src/utils/mos_files_test.txt", "a+") as f:
        for i, item in data.iterrows():
            path = os.path.join("/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/sogun/AfriSpeech-TTS/afrispeech_16k_denoised",
                                item.audio_paths[1:])
            if not os.path.exists(path): continue
            mos_score = wvmos_model.calculate_one(path)
            print(f"{item.audio_ids}|{mos_score}", file=f)
            
    print("finished")