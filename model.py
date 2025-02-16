import numpy as np
import torch
import whisper
import librosa
import soundfile as sf
from gtts import gTTS
import os

# Load pre-trained Whisper model for STT (you could also experiment with a larger model)
whisper_model = whisper.load_model("medium")

def anonymize(input_audio_path):  # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`, 
        ensuring compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """
    audio, sr = librosa.load(input_audio_path, sr=None)
    result = whisper_model.transcribe(input_audio_path, language='en')
    text = result["text"].strip()
    tts_output_path = "anonymized_temp.mp3"
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(tts_output_path)
    anonymized_audio, sr = librosa.load(tts_output_path, sr=None)
    anonymized_audio = librosa.util.normalize(anonymized_audio)
    anonymized_audio = librosa.effects.pitch_shift(anonymized_audio, sr=sr, n_steps=3)
    os.remove(tts_output_path)
    return anonymized_audio.astype(np.float32), sr
