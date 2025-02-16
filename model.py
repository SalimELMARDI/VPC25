import numpy as np
import torch
import whisper
import librosa
import soundfile as sf
from gtts import gTTS
import os

# Load pre-trained Whisper model for STT (you could also experiment with a larger model)
whisper_model = whisper.load_model("small")

def anonymize(input_audio_path):  # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    Anonymization algorithm using STT and TTS.

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

    # Step 1: Load the source audio
    audio, sr = librosa.load(input_audio_path, sr=None)

    # Step 2: Transcribe the audio using Whisper (STT) with explicit language specification
    result = whisper_model.transcribe(input_audio_path, language='en')
    text = result["text"].strip()

    # Step 3: Generate a new voice using Google TTS (gTTS)
    # Use .mp3 extension to match gTTS default output format
    tts_output_path = "anonymized_temp.mp3"
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(tts_output_path)

    # Step 4: Load the new TTS-generated audio (gTTS produces MP3 files)
    anonymized_audio, sr = librosa.load(tts_output_path, sr=None)

    # Optional: Normalize the audio to maintain consistent volume levels and reduce artifacts
    anonymized_audio = librosa.util.normalize(anonymized_audio)

    # Step 5: Apply a more moderate voice transformation (pitch shift)
    # Using n_steps=2 (instead of 4) helps preserve natural enunciation while anonymizing.
    anonymized_audio = librosa.effects.pitch_shift(anonymized_audio, sr=sr, n_steps=2)

    # Clean up temporary file
    os.remove(tts_output_path)

    return anonymized_audio.astype(np.float32), sr
