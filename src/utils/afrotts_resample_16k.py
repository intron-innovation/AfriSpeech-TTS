import os
import librosa
import soundfile as sf

import pandas as pd

import multiprocessing
from multiprocessing import Value, Process


def resample_and_save(data, dir_path):

    for i, item in data.iterrows():
        
        path = item.audio_paths
        src_path = os.path.join(dir_path, "..", item.audio_paths[1:])
        dest_path = os.path.join(dir_path, "afrispeech_16k", f"{item.audio_ids}.wav")
        
        if os.path.isfile(dest_path):
            continue
        
        try:
            audio, sr = librosa.load(src_path, sr=16000)
        except Exception:
            print(src_path)
            continue
            
        sf.write(file=dest_path, data=audio, samplerate=sr, format=None)

    print("finished..")


if __name__ == "__main__":
    
    # path to AfriSpeech dir
    dir_path = os.getcwd()
    
    train = pd.read_csv(os.path.join(dir_path, "data/intron-tts-train-public-28565.csv"))
    dev = pd.read_csv(os.path.join(dir_path, "data/intron-tts-dev-public-3330.csv"))
    test = pd.read_csv(os.path.join(dir_path, "data/intron-tts-test-public-4161.csv"))

    data = pd.concat([train, dev, test])
    
    num_cpu = 16
    
    # start mp computation
    processes = []
    total_splits = len(data)//num_cpu
            
    for i in range(num_cpu-1):
        data_temp = data[total_splits*i: total_splits*i+total_splits]
        # create a child process process
        process = Process(target=resample_and_save, args=(data_temp, dir_path,))
        
        processes.append(process)
        process.start()

    final_split = total_splits*(num_cpu-1)
    data_temp = data[final_split: ]
    process = Process(target=resample_and_save, args=(data_temp, dir_path,))
        
    processes.append(process)
    process.start()

    # complete the processes
    for proc in processes:
        proc.join()
    print("All data has been resampled to 16k !!")
    

# files missing from the dataset
# /AfriSpeech-TTS/test/965756d8-72df-4570-9f9a-136c6aee6eaf/274d814a07965f4cd8e9655625a4e43e_pJ2UcROg.wav
# /AfriSpeech-TTS/test/f29bb76f-49bd-4b29-9164-d0cd2de12071/2f38ad4d51cfe046750c90bb4ceb7e24_iRR8COgR.wav
# /AfriSpeech-TTS/train/9b1a7865-4b74-486e-8223-9117e2ea592a/9b1f7d24a96824967ed3a1ae5d3c44cd_OWVB6wln.wav