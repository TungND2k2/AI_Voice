import os
import subprocess
import torchaudio
import soundfile
import numpy as np
from glob import glob
from loguru import logger
from huggingface_hub import snapshot_download

from viettts.utils.vad import get_speech

import torchaudio
import os
import subprocess
import tempfile


def convert_to_wav(input_filepath: str, target_sr: int) -> str:
    """
    Convert an input audio file to WAV format with the desired sample rate using FFmpeg.

    Args:
        input_filepath (str): Path to the input audio file.
        target_sr (int): Target sample rate.

    Returns:
        str: Path to the converted WAV file.
    """
    temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav_filepath = temp_wav_file.name
    temp_wav_file.close()

    ffmpeg_command = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", input_filepath,
        "-ar", str(target_sr),
        "-ac", "1",
        temp_wav_filepath
    ]

    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        os.unlink(temp_wav_filepath)
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.decode()}")

    return temp_wav_filepath


def load_wav(filepath: str, target_sr: int):
    """
    Load an audio file in any supported format, convert it to WAV, and load as a tensor.

    Args:
        filepath (str): Path to the audio file in any format.
        target_sr (int): Target sample rate.

    Returns:
        Tensor: Loaded audio tensor resampled to the target sample rate.
    """
    # Check if the file is already in WAV format
    if not filepath.lower().endswith(".wav"):
        logger.info(f"Converting {filepath} to WAV format")
        filepath = convert_to_wav(filepath, target_sr)

    # Load the WAV file
    speech, sample_rate = torchaudio.load(filepath)
    speech = speech.mean(dim=0, keepdim=True)  # Convert to mono if not already
    if sample_rate != target_sr:
        assert sample_rate > target_sr, f'WAV sample rate {sample_rate} must be greater than {target_sr}'
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)

    return speech


def save_wav(wav: np.ndarray, sr: int, filepath: str):
    soundfile.write(filepath, wav, sr)


def load_prompt_speech_from_file(filepath: str, min_duration: float=3, max_duration: float=5, return_numpy: bool=False):
    wav = load_wav(filepath, 16000)

    if wav.abs().max() > 0.9:
        wav = wav / wav.abs().max() * 0.9

    wav = get_speech(
        audio_input=wav.squeeze(0),
        min_duration=min_duration,
        max_duration=max_duration,
        return_numpy=return_numpy
    )
    return wav


def load_voices(voice_dir: str):
    files = glob(os.path.join(voice_dir, '*.wav')) + glob(os.path.join(voice_dir, '*.mp3'))
    voice_name_map = {
        os.path.basename(f).split('.')[0]: f
        for f in files
    }
    return voice_name_map


def download_model(save_dir: str):
    snapshot_download(repo_id="dangvansam/viet-tts", local_dir=save_dir)