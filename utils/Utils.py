import torch
import yaml
from typing import Dict, Any
import os
import importlib.metadata
import subprocess

import sys
from typing import Dict, Optional
# Dictionary of required packages and versions
required_versions = {
    'torch': '2.4.1',
    'torchvision': '0.19.1',
    'torchaudio': '2.4.1',
    'xformers': '0.0.25',
    'unsloth': None,  # None means just check if installed
    'accelerate': None,  # Will check latest compatible version
    'huggingface_hub': None,
    'sentence-transformers': '2.7.0',
    'numpy': '1.24.0',
    'transformers': '4.36.2'
}


def check_and_install_packages(required_versions: Dict[str, Optional[str]]):
    """Check and install packages with individual installation attempts."""
    to_install = []
    
    # First check all versions
    for package, required_version in required_versions.items():
        try:
            installed_version = importlib.metadata.version(package)
            
            if required_version and installed_version != required_version:
                print(f"❌ {package} version mismatch (installed: {installed_version}, required: {required_version})")
                to_install.append((package, required_version))
            else:
                print(f"✅ {package} version correct ({installed_version})")
                
        except importlib.metadata.PackageNotFoundError:
            print(f"❌ {package} not installed")
            to_install.append((package, required_version))
    
    if not to_install:
        print("\nAll packages are correctly installed!")
        return
    
    print("\nInstalling missing/incorrect packages...")
    
    # Install packages one by one with error handling
    for package, version in to_install:
        try:
            package_spec = f"{package}=={version}" if version else package
            print(f"Installing {package_spec}...")
            
            # Base command
            cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall"]
            
            # Add CUDA index URL for torch packages
            # if package in ['torch', 'torchvision', 'torchaudio']:
                # cmd.extend(["--index-url", "https://download.pytorch.org/whl/cu124"])
            
            cmd.append(package_spec)
            
            subprocess.run(cmd, check=True)
            print(f"Successfully installed {package_spec}")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_spec}: {e}")
            print("Trying with --no-deps as fallback...")
            try:
                subprocess.run(cmd + ["--no-deps"], check=True)
                print(f"Installed {package_spec} with --no-deps")
            except subprocess.CalledProcessError:
                print(f"❌ Critical: Failed to install {package_spec} even with --no-deps")
    
    print("\nInstallation attempts complete. Some packages may need manual installation.")
    print("Please restart your runtime/kernel if any packages were installed!")

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
    return os.path.join(audio_dir, audio_file)