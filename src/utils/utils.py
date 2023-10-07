import argparse


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csv_path",
        type=str,
        default="./data/intron-dev-tiny-public-25-clean.csv",
        help="path to data csv file",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="./data/",
        help="directory to locate the audio",
    )
    parser.add_argument(
        "--model_id_or_path",
        type=str,
        default="facebook/mms-tts-eng",
        help="id of the whisper model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="directory to store results"
    )

    return parser.parse_args()
