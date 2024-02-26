import os
import gc
import sys
import argparse
import configparser
import random
import subprocess
import time
import warnings
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
from datasets import load_metric
import transformers
import datasets
from transformers import (
    Seq2SeqTrainingArguments,
    AdamW,
    Seq2SeqTrainer,
    SpeechT5ForTextToSpeech

)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from src.utils.prepare_dataset import DataConfig, data_prep, get_speaker_model, TTSDataCollatorWithPadding
import logging


data_home = "data3"
os.environ['HF_HOME'] = f'/{data_home}/.cache/'
os.environ['XDG_CACHE_HOME'] = f'/{data_home}/.cache/'
os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.DEBUG)

gc.collect()
torch.cuda.empty_cache()

# warnings.filterwarnings('ignore')
SAMPLING_RATE = 16000
PROCESSOR = None

num_of_gpus = torch.cuda.device_count()
print("num_of_gpus:", num_of_gpus)
print("torch.cuda.is_available()", torch.cuda.is_available())
print("cuda.current_device:", torch.cuda.current_device())
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


def parse_argument():
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser(prog="Train")
    parser.add_argument("-c", "--config", dest="config_file",
                        help="Pass a training config file", metavar="FILE")
    parser.add_argument("--local_rank", type=int,
                        default=0)
    parser.add_argument("-g", "-gpu", "--gpu", type=int,
                        default=0)
    args = parser.parse_args()
    config.read(args.config_file)
    return args, config


def train_setup(config, args):
    repo_root = config['experiment']['repo_root']
    exp_dir = os.path.join(repo_root, config['experiment']['dir'], config['experiment']['name'])
    config['experiment']['dir'] = exp_dir
    checkpoints_path = os.path.join(exp_dir, 'checkpoints')
    config['checkpoints']['checkpoints_path'] = checkpoints_path
    figure_path = os.path.join(exp_dir, 'figures')
    config['logs']['figure_path'] = figure_path
    predictions_path = os.path.join(exp_dir, 'predictions')
    config['logs']['predictions_path'] = predictions_path

    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    subprocess.call(['cp', args.config_file, f"{exp_dir}/{args.config_file.split('/')[-1]}"])
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    Path(figure_path).mkdir(parents=True, exist_ok=True)
    Path(predictions_path).mkdir(parents=True, exist_ok=True)

    print(f"using exp_dir: {exp_dir}. Starting...")

    return checkpoints_path


def data_setup(config):
    data_config = DataConfig(
        train_path=config['data']['train'],
        val_path=config['data']['val'],
        aug_path=config['data']['aug'] if 'aug' in config['data'] else None,
        aug_percent=float(config['data']['aug_percent']) if 'aug_percent' in config['data'] else None,
        exp_dir=config['experiment']['dir'],
        ckpt_path=config['checkpoints']['checkpoints_path'],
        model_path=config['models']['model_path'],
        spk_model_name=config['models']['spk_model_name'],
        audio_path=config['audio']['audio_path'],
        max_audio_len_secs=int(config['hyperparameters']['max_audio_len_secs']),
        min_transcript_len=int(config['hyperparameters']['min_transcript_len']),
        max_transcript_len=int(config['hyperparameters']['max_label_len']),
        domain=config['data']['domain'],
        seed=int(config['hyperparameters']['data_seed']),
        reduction_factor=int(config['hyperparameters']['reduction_factor']),
    )
    return data_config



def get_data_collator(config):
    speaker_model = get_speaker_model(config.spk_model_name)

    data_collator_ = TTSDataCollatorWithPadding(processor=PROCESSOR, speaker_model=speaker_model)
    return data_collator_


def get_checkpoint(checkpoint_path, model_path):
    last_checkpoint_ = None

    ckpt_files = os.listdir(checkpoint_path)
    if "pytorch_model.bin" in ckpt_files:
        return checkpoint_path, checkpoint_path

    if os.path.isdir(checkpoint_path):
        last_checkpoint_ = get_last_checkpoint(checkpoint_path)
        if last_checkpoint_ is None and len(os.listdir(checkpoint_path)) > 0:
            logging.warn(
                f"Output directory ({checkpoint_path}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint_ is not None:
            logging.warn(
                f"Checkpoint detected, resuming training at {last_checkpoint_}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # use last checkpoint if exist 
    if last_checkpoint_:
        checkpoint = last_checkpoint_
    elif os.path.isdir(model_path):
        checkpoint = None
    else:
        checkpoint = None

    return last_checkpoint_, checkpoint


def set_dropout(trained_model):
    trained_model.eval()
    for name, module in trained_model.named_modules():
        if 'dropout' in name:
            module.train()


if __name__ == "__main__":

    args, config = parse_argument()

    checkpoints_path = train_setup(config, args)
    data_config = data_setup(config)
    if is_main_process:
        train_dataset, val_dataset, aug_dataset, PROCESSOR = data_prep(data_config)
    data_collator = get_data_collator(data_config)

    start = time.time()
    # Detecting last checkpoint.
    last_checkpoint, checkpoint_ = get_checkpoint(checkpoints_path, config['models']['model_path'])
    model = SpeechT5ForTextToSpeech.from_pretrained(
        last_checkpoint if last_checkpoint else config['models']['model_path'],
        pad_token_id=PROCESSOR.tokenizer.pad_token_id,
    )
    if config['hyperparameters']['gradient_checkpointing'] == "True":
        model.gradient_checkpointing_enable()

    print(f"\n...Model loaded in {time.time() - start:.4f}.\n")

    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoints_path,
        overwrite_output_dir=True if config['hyperparameters']['overwrite_output_dir'] == "True" else False,
        group_by_length=True if config['hyperparameters']['group_by_length'] == "True" else False,
        data_seed=int(config['hyperparameters']['data_seed']),
        per_device_train_batch_size=int(config['hyperparameters']['train_batch_size']),
        per_device_eval_batch_size=int(config['hyperparameters']['val_batch_size']),
        gradient_accumulation_steps=int(config['hyperparameters']['gradient_accumulation_steps']),
        gradient_checkpointing=True if config['hyperparameters']['gradient_checkpointing'] == "True" else False,
        ddp_find_unused_parameters=True if config['hyperparameters']['ddp_find_unused_parameters'] == "True" else False,
        evaluation_strategy="steps",
        max_steps=int(config['hyperparameters']['max_steps']),
        fp16=torch.cuda.is_available(),
        save_steps=int(config['hyperparameters']['save_steps']),
        eval_steps=int(config['hyperparameters']['eval_steps']),
        logging_steps=int(config['hyperparameters']['logging_steps']),
        learning_rate=float(config['hyperparameters']['learning_rate']),
        warmup_ratio=float(config['hyperparameters']['warmup_ratio']),
        max_grad_norm=float(config['hyperparameters']['max_grad_norm']),
        save_total_limit=int(config['hyperparameters']['save_total_limit']),
        dataloader_num_workers=int(config['hyperparameters']['dataloader_num_workers']),
        logging_first_step=True,
        log_level='debug',
        load_best_model_at_end=True if config['hyperparameters']['load_best_model_at_end'] == 'True' else False,
        greater_is_better=False,
        ignore_data_skip=True if config['hyperparameters']['ignore_data_skip'] == 'True' else False,
        label_names=["labels"],
        report_to=None
    )

    # set the main code and the modules it uses to the same log-level according to the node
    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity_debug()

    print("Training Args:\n", training_args.__dict__)

    print("device: ", training_args.device, device)

    print(f"\n...Model Args loaded in {time.time() - start:.4f}. Start training...\n")
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=PROCESSOR.tokenizer,
    )

    if config['hyperparameters']['do_train'] == "True":
        PROCESSOR.save_pretrained(checkpoints_path)

        trainer.train(resume_from_checkpoint=checkpoint_)

        model.save_pretrained(checkpoints_path)
        PROCESSOR.save_pretrained(checkpoints_path)

    if config['hyperparameters']['do_eval'] == "True":
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(val_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)