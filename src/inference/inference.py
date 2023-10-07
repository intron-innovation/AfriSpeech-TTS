import os
from transformers import VitsModel, AutoTokenizer
import torch
import scipy

from src.utils.utils import parse_argument


def afri_speak(model_id, output_dir, output_file_name="techno.wav"):
    model = VitsModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = "some example text in the English language"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    output_path = os.path.join(output_dir, output_file_name)
    scipy.io.wavfile.write(output_path, rate=model.config.sampling_rate, data=output.float().numpy())


if __name__ == "__main__":
    """Run main script"""

    args = parse_argument()

    # Make output directory if does not already exist
    os.makedirs(args.output_dir, exist_ok=True)

    afri_speak(args.model_id_or_path, args.output_dir)
