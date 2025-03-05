import os

import torch

import torchaudio
from torchaudio import transforms

import tqdm

import pandas as pd

from SpeechCARE_Linguistic_Explainability_Framework.utils.Utils import report

def get_audio_path(uid, audio_dir):
    audio_file = f"{uid}.wav"
    return os.path.join(audio_dir, audio_file)

def add_label(df):

    def determine_label(row):
        if row["diagnosis_control"] == 1.0:
            return 0
        elif row["diagnosis_mci"] == 1.0:
            return 1
        elif row["diagnosis_adrd"] == 1.0:
            return 2
        return -1  # Optional: Handle unexpected cases

    df["label"] = df.apply(determine_label, axis=1)

    df = df.drop(columns=["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"])

    return df

def segment_audios_to_tensors(
    
    df,
    output_dir,
    segment_length=5,
    min_acceptable=5,
    target_sr=16000,
    overlap=0.2
    ):

    results = []
    os.makedirs(output_dir, exist_ok=True)

    include_label = "label" in df.columns  # Check if label column exists

    for idx, row in tqdm(df.iterrows(), desc="Processing audio files", total=len(df)):
        uid = row["uid"]
        audio_path = row["path"]
        label = row["label"] if include_label else None  # Retrieve label if it exists

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
        segment_samples = segment_length * target_sr
        overlap_samples = int(segment_samples * overlap)
        step_samples = segment_samples - overlap_samples
        num_segments = (int(audio.size(0)) - segment_samples) // step_samples + 1

        segments = []
        end_sample = 0

        # Create segments
        for i in range(num_segments):
            start_sample = i * step_samples
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            segments.append(segment)

        remaining_part = audio[end_sample:]
        if remaining_part.size(0) >= min_acceptable * target_sr:
            segments.append(remaining_part)

        # Save segments as tensors
        uid_dir = os.path.join(output_dir, uid)
        os.makedirs(uid_dir, exist_ok=True)

        for i, seg in enumerate(segments):
            tensor_path = os.path.join(uid_dir, f"{uid}_segment_{i}.pt")
            torch.save(seg, tensor_path)

            # Append result to the list
            result = {
                "uid": uid,
                "segment": i,
                "path": tensor_path,
            }
            if include_label:  # Add label only if it exists
                result["label"] = label

            results.append(result)

    # Create and return the result dataframe
    result_df = pd.DataFrame(results)

    max_num_segments = calculate_num_segments(30, segment_length, overlap, min_acceptable)

    return result_df, max_num_segments

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

def get_dataset(config, train_path, valid_path = None, test_path = None,
                train_audio_path=None, valid_audio_path = None, test_audio_path = None):

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # add audio path to subjects dataframes
    train_df["path"] = train_df["uid"].apply(lambda uid: get_audio_path(uid, train_audio_path))
    valid_df["path"] = valid_df["uid"].apply(lambda uid: get_audio_path(uid, valid_audio_path))
    test_df["path"] = test_df["uid"].apply(lambda uid: get_audio_path(uid, test_audio_path))

    # add label column and delete the dignoisis columns
    train_df = add_label(train_df)
    valid_df = add_label(valid_df)

    train_df = train_df[train_df.valid == 'Yes']
    valid_df = valid_df[valid_df.valid == 'Yes']

    # resample and segment audio files
    valid_segments_df, _ = segment_audios_to_tensors(valid_df, 'audio_segments')
    test_segments_df, _ = segment_audios_to_tensors(test_df, 'audio_segments')
    train_segments_df, max_num_segments = segment_audios_to_tensors(train_df, 'audio_segments')


    config.max_num_segments = max_num_segments
    report(f'\nMax number of segments: {max_num_segments}', True)

    dataset = {'TRAIN': {'SEGMENTS': train_segments_df, 'TRANSCRIP': train_df},
            'VALID': {'SEGMENTS': valid_segments_df, 'TRANSCRIP': valid_df} ,
            'TEST': {'SEGMENTS': test_segments_df, 'TRANSCRIP': test_df},
            }
    return dataset