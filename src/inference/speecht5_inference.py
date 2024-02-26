import os
from typing import List, Dict, Union, Set
from transformers import pipeline, VitsModel, AutoTokenizer
import torch
import scipy
from datasets import load_dataset
import soundfile as sf
import librosa
import pandas as pd
from src.utils.utils import parse_argument
from tqdm import tqdm
from src.utils.prepare_dataset import get_speaker_model, create_speaker_embedding, load_afri_speech_data

device = "cuda" if torch.cuda.is_available() else "cpu"



def generate_speech(text:str, synthesiser, speaker_embedding) ->  Dict:
    speech_array = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    return speech_array


def run_inference(args, dataset, synthesiser, speaker_model):
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text = row['transcript']
        audio_path = row['audio_paths']
        audio_array, sr = librosa.load(audio_path)
        audio_len = len(audio_array)/ 1600
        print("print audio len is: ", audio_len)
        max_audio_len = audio_len + 20
        print("print max_audio_len is: ", max_audio_len)

        audio_embedding = create_speaker_embedding(speaker_model, audio_array)
        audio_embedding = torch.tensor(audio_embedding).unsqueeze(0)
        count_in_loop = 0
        while max_audio_len >= audio_len:
            gen_speech = generate_speech(text, synthesiser, audio_embedding)
            max_audio_len = len(gen_speech['audio']) / gen_speech['sampling_rate']
            count_in_loop +=1
            print(count_in_loop)
            if count_in_loop == 5:
                count_in_loop = 0
                break
        
        output_path = os.path.join(args.output_dir, audio_path.split("/")[-1])
        sf.write(output_path, gen_speech["audio"], samplerate=gen_speech["sampling_rate"])
        dataset.loc[index, 'gen_path'] = output_path
        dataset.loc[index, 's3_gen_path'] = output_path.replace(args.audio_dir, "http://intron-open-source.s3.amazonaws.com/AfriSpeech-TTS-D/")

        
    return dataset


def main():
    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)


    synthesiser = pipeline("text-to-speech", args.model_id_or_path)
    speaker_model = get_speaker_model(args.speaker_model_id_or_path)

    dataset = load_afri_speech_data(args.data_csv_path, audio_dir=args.audio_dir, return_dataset=False)
    pred_df = run_inference(args, dataset, synthesiser, speaker_model)
    pred_df.to_csv(f"./results/{args.data_csv_path.replace('/', '-')}_preds.csv", index=False)




if __name__ == "__main__":
    main()

#fintuned    
# python3 src/inference/speecht5_inference.py --data_csv_path  ./data/afritts-test-seen-clean.csv --audio_dir /data3/data/AfriSpeech-TTS/ --model_id_or_path /data3/abraham/tts/AfriSpeech-TTS/src/experiments/afri_tts_speech_t5_denoised/checkpoints/checkpoint-19500  --output_dir /data3/data/AfriSpeech-TTS/tts_generated_speech/afritts_test_seen/speech_t5_finetuned
# python3 src/inference/speecht5_inference.py --data_csv_path  ./data/afritts-test-unseen-clean.csv --audio_dir /data3/data/AfriSpeech-TTS/ --model_id_or_path /data3/abraham/tts/AfriSpeech-TTS/src/experiments/afri_tts_speech_t5_denoised/checkpoints/checkpoint-19500  --output_dir /data3/data/AfriSpeech-TTS/tts_generated_speech/afritts_test_unseen/speech_t5_finetuned

#baseline    
# python3 src/inference/speecht5_inference.py --data_csv_path  ./data/afritts-test-seen-clean.csv --audio_dir /data3/data/AfriSpeech-TTS/ --model_id_or_path microsoft/speecht5_tts  --output_dir /data3/data/AfriSpeech-TTS/tts_generated_speech/afritts_test_seen/speech_t5_baseline
# python3 src/inference/speecht5_inference.py --data_csv_path  ./data/afritts-test-unseen-clean.csv --audio_dir /data3/data/AfriSpeech-TTS/ --model_id_or_path microsoft/speecht5_tts  --output_dir /data3/data/AfriSpeech-TTS/tts_generated_speech/afritts_test_unseen/speech_t5_baseline