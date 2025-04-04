import os
import gdown
import torch
from pathlib import Path
from typing import Optional
import hashlib

class WeightManager:
    def __init__(self, weights_dir: str = "model_weights"):
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)
        
    def download_weights(
        self,
        file_id: str,
        output_name: str,
        md5_hash: Optional[str] = None,
        force_redownload: bool = False
    ) -> str:
        """
        Download weights from Google Drive with verification
        
        Args:
            file_id: Google Drive file ID (from shareable link)
            output_name: Name for downloaded file
            md5_hash: Expected MD5 hash for verification (optional)
            force_redownload: Whether to download even if file exists
            
        Returns:
            Path to downloaded weights file
            
        Example:
            >>> manager = WeightManager()
            >>> path = manager.download_weights(
                    file_id="1a2b3c4d5e6f7g8h9i0j",
                    output_name="hubert_base.pt",
                    md5_hash="a1b2c3d4e5f6g7h8i9j0"
                )
        """
        output_path = self.weights_dir / output_name
        
        # Skip if exists and valid
        if not force_redownload and output_path.exists():
            if md5_hash is None or self._verify_md5(output_path, md5_hash):
                return str(output_path)
        
        # Download with progress
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
        
        # Verify download
        if md5_hash and not self._verify_md5(output_path, md5_hash):
            os.remove(output_path)
            raise ValueError("Downloaded file hash doesn't match expected!")
            
        return str(output_path)
    
    def _verify_md5(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file MD5 hash matches expected"""
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash == expected_hash
    
    def load_torch_weights(self, file_id: str, output_name: str, **kwargs):
        """Download and load weights directly into memory"""
        weight_path = self.download_weights(file_id, output_name, **kwargs)
        return torch.load(weight_path, map_location='cpu')