import argparse


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="./data/afritts-test-seen-clean.csv",
        help="path to data csv file",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/data3/data/AfriSpeech-TTS-D/",
        help="directory to locate the audio",
    )
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="microsoft/speecht5_tts",
        help="id of the speech model model",
    )
    parser.add_argument(
        "--speaker_model_id_or_path",
        type=str,
        default="speechbrain/spkrec-xvect-voxceleb",
        help="id of the speaker model to get embeddings",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./audios", help="directory to store results"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="My name is Abraham Owodunni, and my native language is Yoruba",
        help="sample text to speak",
    )

    return parser.parse_args()
