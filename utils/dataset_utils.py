import torch
import torchaudio
import torchaudio.transforms as transforms 

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from sklearn.preprocessing import OneHotEncoder

import os
import numpy as np

import pandas as pd

import scipy.signal as signal

from utils.Utils import get_audio_path

def calculate_num_segments(audio_duration, segment_length, overlap, min_acceptable):
    """
    Calculate the maximum number of segments for a given audio duration.

    Args:
        audio_duration (float): Total duration of the audio in seconds.
        segment_length (float): Length of each segment in seconds.
        overlap (float): Overlap between consecutive segments as a fraction of segment length.
        min_acceptable (float): Minimum length of the remaining part to be considered a segment.

    Returns:
        int: Maximum number of segments.
    """
    overlap_samples = segment_length * overlap
    step_samples = segment_length - overlap_samples
    num_segments = int((audio_duration - segment_length) // step_samples + 1)
    remaining_audio = (audio_duration - segment_length) - (step_samples * (num_segments - 1))
    if remaining_audio >= min_acceptable:
        num_segments += 1
    return num_segments


def preprocess_audio(audio_path, segment_length=5, overlap=0.2, target_sr=16000):
        """
        Preprocess a single audio file into segments.

        Args:
            audio_path (str): Path to the input audio file.
            segment_length (int): Length of each segment in seconds.
            overlap (float): Overlap between consecutive segments as a fraction of segment length.
            target_sr (int): Target sampling rate for resampling.

        Returns:
            torch.Tensor: Tensor containing the segmented audio data.
        """
        # Load and resample audio
        audio, sr = torchaudio.load(audio_path)
        resampler = transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)

        # Convert to mono (average across channels)
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0)  # Average across channels
        else:
            audio = audio.squeeze(0)

        # Calculate segment parameters
        segment_samples = int(segment_length * target_sr)
        overlap_samples = int(segment_samples * overlap)
        step_samples = segment_samples - overlap_samples
        num_segments = calculate_num_segments(len(audio) / target_sr, segment_length, overlap, 5)
        segments = []
        end_sample = 0

        # Create segments
        for i in range(num_segments):
            start_sample = i * step_samples
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            segments.append(segment)

        # Handle remaining part
        remaining_part = audio[end_sample:]
        if len(remaining_part) >= 5 * target_sr:
            segments.append(remaining_part)

        # Stack segments into a tensor
        waveform = torch.stack(segments)  # Shape: [num_segments, seq_length]
        return waveform.unsqueeze(0)  # Add batch dimension: [1, num_segments, seq_length]


def lowpass(waveform, sampling_rate, cutoff_freq=3000, order=5):
    waveform = waveform.numpy()
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_waveform = signal.lfilter(b, a, waveform, axis=1)
    return filtered_waveform

def lpfilter_audio_files(audio_path, output_dir):
    assert audio_path is not None, "File path cannot be None"
    assert os.path.exists(audio_path), f"File not found: {audio_path}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, audio_path.split("/")[-1].split(".")[0] + ".wav")
    try:
        noisy, sr = torchaudio.load(audio_path)
        # Trim to 30 seconds
        max_length = sr * 30  # Number of samples in 30 seconds
        noisy = noisy[:, :max_length]  # Trim if longer than 30s
        
        filtered_waveform = torch.tensor(lowpass(noisy, sr, 8000, 5))
        torchaudio.save(output_path, filtered_waveform, sr)
        return output_path
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def get_whisper_model():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def get_whisper_transcription_and_lang(audio_path, pipe):

    assert audio_path is not None, "File path cannot be None"
    assert os.path.exists(audio_path), f"File not found: {audio_path}"

    # Load and resample audio
    audio, sr = torchaudio.load(audio_path,format="wav")
    
    resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
    audio = resampler(audio)

    # Convert to mono (average across channels)
    if audio.size(0) > 1:
        audio = torch.mean(audio, dim=0)  # Average across channels
    else:
        audio = audio.squeeze(0)

    audio = np.array(audio)


    try:
        pipe = get_whisper_model()
        result = pipe(audio)
            
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    
    
    return result['text']

def get_age_category(age: int) -> str:
    """
    Categorize a single age value into predefined bins.
    
    Args:
        age: Integer age value to categorize
        
    Returns:
        str: Age category string
    """
    # Create categories of age by mapping the age rages to "0","1","2"
    demography_mapping = {"40-65": 0, "66-80": 1, "+80": 2}
    if age < 40:
        print("Invalid age input")
        return None
    elif 40 <= age < 66:
        return demography_mapping["40-65"]
    elif 66 <= age < 81:
        return demography_mapping["66-80"] 
    else:
        return demography_mapping["+80"] 

def preprocess_data(audio_path: str,
               age: int,
               output_dir: str = "./processed_audio") -> pd.DataFrame:
    """
    Prepares the dataframe by:
    1. Validating required columns (uid, age)
    2. Generating audio paths
    3. Applying low-pass filtering to audio files
    4. Adding age bins
    5. Generating transcriptions
    
    Args:
        df: Input dataframe with columns (uid, age) and optionally gender
        output_dir: Directory to save processed audio files (default: './processed_audio')
        
    Returns:
        Processed dataframe with additional columns:
        - processed_audio_path: Path to filtered audio file
        - age_bin: Categorical age bin
        - transcription: Whisper-generated transcription
    """
    

    processed_audio_path = lpfilter_audio_files(audio_path, output_dir)

    age_category = get_age_category(age)
    
    # 5. Generate transcriptions (load model once for efficiency)
    pipe = get_whisper_model()
    transcription = get_whisper_transcription_and_lang(processed_audio_path, pipe)

    
    return processed_audio_path, age_category,transcription