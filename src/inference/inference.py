import os
from transformers import pipeline, VitsModel, AutoTokenizer
import torch
import scipy
from datasets import load_dataset
import soundfile as sf

from src.utils.utils import parse_argument


def afri_vits_speak(model_id, output_dir, text, output_file_name="vits.wav"):
    """
    https://huggingface.co/facebook/mms-tts-eng

    VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) is an end-to-end speech
    synthesis model that predicts a speech waveform conditional on an input text sequence.
    It is a conditional variational autoencoder (VAE) comprised of a posterior encoder, decoder, and conditional prior.

    :param model_id: str
    :param output_dir: str
    :param text: str
    :param output_file_name: str
    :return: output file path, str
    """

    model = VitsModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    output = torch.t(output)

    output_path = os.path.join(output_dir, output_file_name)
    scipy.io.wavfile.write(output_path, rate=model.config.sampling_rate, data=output.float().numpy())
    return output_path


def afri_speecht5(model_id, output_dir, text, output_file_name="tt5.wav",
                  speakers_dataset="Matthijs/cmu-arctic-xvectors",
                  speaker_id=7306,
                  ):
    """
    https://huggingface.co/microsoft/speecht5_tts

    unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text
    representation learning. The SpeechT5 framework consists of a shared encoder-decoder
    network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input
    speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation,
    and then the post-nets generate the output in the speech/text modality based on the output of the decoder.

    :param model_id: str, e.g. microsoft/speecht5_tts
    :param output_dir: str
    :param text: str
    :param output_file_name: str
    :param speakers_dataset: str
    :param speaker_id: int
    :return: str, output path
    """
    synthesiser = pipeline("text-to-speech", model_id)

    embeddings_dataset = load_dataset(speakers_dataset, split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})

    output_path = os.path.join(output_dir, output_file_name)

    sf.write(output_path, speech["audio"], samplerate=speech["sampling_rate"])
    return output_path


if __name__ == "__main__":
    """Run main script"""

    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)

    if "speecht5" in args.model_id_or_path:
        afri_speecht5(args.model_id_or_path, args.output_dir, args.text)
    elif "mms-tts" in args.model_id_or_path:
        afri_vits_speak(args.model_id_or_path, args.output_dir, args.text)
    else:
        raise NotImplementedError(f"{args.model_id_or_path} not supported")
