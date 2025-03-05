import torch
import torchaudio
import torchaudio.transforms as transforms 

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