import logging
import os
import time
import sys
import pandas as pd
import os
import torch
from speechbrain.pretrained import EncoderClassifier
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from transformers import SpeechT5Processor


from src.utils.audio_processing import AudioConfig, load_audio_file
from src.utils.text_processing import clean_text


os.environ['TRANSFORMERS_CACHE'] = '/data3/.cache/'
os.environ['XDG_CACHE_HOME'] = '/data3/.cache/'


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging_level = logging.DEBUG
logger.setLevel(logging_level)

PROCESSOR = None
CONFIG = None
MAX_MODEL_AUDIO_LEN_SECS = 87
LABEL_MAP = {}


device = "cuda" if torch.cuda.is_available() else "cpu"

class DataConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_afri_speech_data(
        data_path, max_audio_len_secs=30, audio_dir=f"./data/",
        return_dataset=True, split="dev", gpu=-1, domain='all',
        max_transcript_len=-1, min_transcript_len=-1
):
    """
    load train/dev/test data from csv path.
    :param max_transcript_len:
    :param min_transcript_len:
    :param domain:
    :param gpu:
    :param split:
    :param return_dataset:
    :param audio_dir:
    :param max_audio_len_secs: int
    :param data_path: str
    :return: Dataset instance
    """
    data = pd.read_csv(data_path)
    data["audio_paths"] = data["audio_paths"].apply(
        lambda x: x.replace("/AfriSpeech-TTS/", audio_dir)
    )
    
    if not os.path.exists(data["audio_paths"].values.tolist()[0]):
        data["audio_paths"] = data["audio_paths"].apply(
        lambda x: x.replace(audio_dir, f"/{audio_dir}/{split}/")
        )

        if not os.path.exists(data["audio_paths"].values.tolist()[0]):
            raise Exception(f'Could not find a way to replace the `audio_dir` path in order to retrieve your audio files. \n audio_dir: {audio_dir}')
    
    if max_audio_len_secs > -1 and gpu != -1:
        # when gpu is available, it cannot fit long samples
        data = data[data.duration < max_audio_len_secs]
    elif gpu == -1 and max_audio_len_secs > MAX_MODEL_AUDIO_LEN_SECS:
        # if cpu, infer all samples, no filtering
        pass
    elif gpu == -1 and max_audio_len_secs != -1:
        # if cpu, infer only long samples
        # assuming gpu has inferred on all short samples
        data = data[data.duration >= max_audio_len_secs]
    else:
        # Check if any of the sample is longer than
        # the GPU global MAX_MODEL_AUDIO_LEN_SECS
        if (gpu != -1) and (data.duration.to_numpy() > MAX_MODEL_AUDIO_LEN_SECS).any():
            raise ValueError(
                f"Detected speech longer than {MAX_MODEL_AUDIO_LEN_SECS} secs"
                "-- set `max_audio_len_secs` to filter longer speech!"
            )
    # breakpoint()
    if domain != 'all':
        data = data[data.domain == domain]
    if min_transcript_len != -1:
        data = data[data.transcript.str.len() >= min_transcript_len]
    if max_transcript_len != -1:
        data = data[data.transcript.str.len() < max_transcript_len]

    # dropping speakers < 100
    #data = data[data['user_ids'].isin(data['user_ids'].value_counts().values >= 10)]

    data["text"] = data["transcript"]
    print("before dedup", data.shape)
    data.drop_duplicates(subset=["audio_paths"], inplace=True)
    print("after dedup", data.shape)
    if return_dataset:
        return Dataset.from_pandas(data)
    else:
        return data


def data_prep(config):
    # Prepare data for the model
    global CONFIG, PROCESSOR
    CONFIG = config
    start = time.time()
    aug_dataset = None

    raw_dataset = load_data(config.train_path, config.val_path, config.aug_path)
    logger.debug(f"...Data Read Complete in {time.time() - start:.4f}. Starting Tokenizer...")


    PROCESSOR = load_processor(config.model_path)
    logger.debug(f"...Load vocab and processor complete in {time.time() - start:.4f}.\n"
                 f"Loading dataset...")

    val_dataset = load_custom_dataset(config, config.val_path, 'dev',
                                      transform_audio, transform_labels,
                                      )
    if config.aug_percent and config.aug_percent > 1:
        train_df = load_custom_dataset(config, config.train_path, 'train',
                                       transform_audio, transform_labels, return_dataset=False,
                                      )
        aug_df = train_df.sample(frac=config.aug_percent, random_state=config.seed)
        train_df = train_df[~train_df.audio_ids.isin(aug_df.audio_ids.to_list())]
        aug_dataset = Dataset.from_pandas(aug_df)
        train_dataset = Dataset.from_pandas(train_df)
    elif config.aug_path:
        train_dataset = load_custom_dataset(config, config.train_path, 'train',
                                            transform_audio, transform_labels,
                                            )
        aug_dataset = load_custom_dataset(config, config.aug_path, 'aug',
                                          transform_audio, transform_labels,
                                          )
    else:
        split = 'train' if 'train' in config.train_path else 'dev'
        train_dataset = load_custom_dataset(config, config.train_path, split,
                                            transform_audio, transform_labels,
                                            )

    logger.debug(f"Load train and val dataset done in {time.time() - start:.4f}.")
    return train_dataset, val_dataset, aug_dataset, PROCESSOR


def load_custom_dataset(config, data_path, split,
                        transform_audio_, transform_labels_=None,
                        prepare=None, return_dataset=True, multi_task=None):
    return CustomASRDataset(config, data_path, transform_audio_, transform_labels_,
                            config.audio_path, split=split, domain=config.domain,
                            max_audio_len_secs=config.max_audio_len_secs,
                            min_transcript_len=config.min_transcript_len,
                            prepare=prepare, return_dataset=return_dataset,
                            multi_task=multi_task)

def load_data(train_path, val_path, aug_path=None):
    if aug_path:
        dataset = load_dataset('csv', data_files={'train': train_path, 'val': val_path, 'aug': aug_path})
    else:
        dataset = load_dataset('csv', data_files={'train': train_path, 'val': val_path})

    return dataset


def load_processor(tokenizer_path):
    processor = SpeechT5Processor.from_pretrained(tokenizer_path)
    return processor

def get_speaker_model(spk_model_name):

    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", spk_model_name)
    )

    return speaker_model

def create_speaker_embedding(speaker_model, speech):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(speech).to(device))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().to("cpu").numpy()
    return speaker_embeddings


def transform_audio(audio_path, text):
    speech = load_audio_file(audio_path)
    
    # feature extraction and tokenization
    data = PROCESSOR(text=text, audio_target=speech, sampling_rate=AudioConfig.sr,  return_attention_mask=False)
    
    # strip off the batch dimensionp
    data["labels"] = data["labels"][0]
    data['array'] = speech
    return data

def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200



def transform_labels(text):
    text = clean_text(text)
    return text




class CustomASRDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_file, transform=None, transform_target=None, audio_dir=None,
                 split=None, domain="all", max_audio_len_secs=-1, min_transcript_len=10,
                 prepare=True, max_transcript_len=-1, gpu=1,
                 length_column_name='duration', return_dataset=True,
                 multi_task=None):

        self.config = config
        self.prepare = prepare
        self.split = split
        self.asr_data = load_afri_speech_data(data_file, min_transcript_len=min_transcript_len,
                                              max_audio_len_secs=max_audio_len_secs,
                                              split=split, gpu=gpu,
                                              audio_dir=audio_dir,
                                              max_transcript_len=max_transcript_len,
                                              domain=domain, return_dataset=return_dataset)
        self.transform = transform
        self.target_transform = transform_target
        self.multi_task = multi_task

    def set_dataset(self, new_data):
        self.asr_data = Dataset.from_pandas(new_data, preserve_index=False)

    def get_dataset(self):
        return self.asr_data.to_pandas()

    def __len__(self):
        return len(self.asr_data)

    def __getitem__(self, idx):
        audio_path = self.asr_data[idx]['audio_paths']
        text = self.target_transform(self.asr_data[idx]['transcript'])
        accent = self.asr_data[idx]['accent']
        audio_idx = self.asr_data[idx]['audio_ids']
        domain = self.asr_data[idx]['domain']
        vad = self.asr_data[idx].get('vad', 'speech')

        input_features = self.transform(audio_path=audio_path, text=text)
        
        if is_not_too_long(input_features):
            return input_features
        
        else:
            del self.asr_data[idx]
            return self.__getitem__(self, idx)


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any
    speaker_model: Any 

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [create_speaker_embedding(self.speaker_model, feature["array"]) for feature in features]


        # collate the inputs and targets into a batch
        batch = PROCESSOR.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if CONFIG.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % CONFIG.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
    
    
