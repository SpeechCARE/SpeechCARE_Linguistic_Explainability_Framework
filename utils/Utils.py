import torch
import yaml
from typing import Dict, Any
import os
def report(text, space = False):
    print(text)
    if space: print('-' * 50)

def free_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def load_yaml_file(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_audio_path(uid: str, audio_dir: str, format:str = '.wav') -> str:
    """Construct full path to audio file from unique ID.
    
    Args:
        uid: Unique identifier for audio file
        audio_dir: Directory containing audio files
        
    Returns:
        Full path to audio file (e.g., '/path/to/audio_dir/uid.mp3')
    """
    audio_file = f"{uid}{format}"
    print(os.path.join(audio_dir, audio_file))
    return os.path.join(audio_dir, audio_file)