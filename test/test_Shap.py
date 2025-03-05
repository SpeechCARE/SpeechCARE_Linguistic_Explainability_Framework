import sys
import argparse
import torch
import yaml
from typing import Optional, Tuple, Dict, Any

 # Add custom paths to sys.path (if needed)
sys.path.append("SpeechCARE_Linguistic_Explainability_Framework")

from SHAP.Shap import LinguisticShap
from models.ModelWrapper import ModelWrapper
from utils.Config import Config

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Test Linguistic Explainability Framework")

    # Required arguments
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the PyTorch model to use as the pretrained model.")
    
    # Optional arguments
    parser.add_argument("--predicted_label", type=int, default=None,
                        help="Precomputed predicted label (optional).")
    parser.add_argument("--transcription", type=str, default=None,
                        help="Precomputed transcription (optional).")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the results (optional).")
    parser.add_argument("--demography_info", type=float, default=None,
                        help="Demographic information of shape [batch_size, 1] (optional).")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Path to the audio file of the selected sample (optional).")

    return parser.parse_args()

def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate that either (audio_path and demography_info) or (predicted_label and transcription) are provided.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Raises:
        ValueError: If neither pair of arguments is provided.
    """
    if not ((args.audio_path is not None and args.demography_info is not None) or 
            (args.predicted_label is not None and args.transcription is not None)):
        raise ValueError("Either 'audio_path' and 'demography_info' or 'predicted_label' and 'transcription' must be provided.")

def initialize_model(config: Dict[str, Any], model_checkpoint: str) -> torch.nn.Module:
    """
    Initialize the model wrapper and load the pretrained model.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the model.
        model_checkpoint (str): Path to the model checkpoint.

    Returns:
        torch.nn.Module: Loaded model.
    """
    wrapper = ModelWrapper(config)
    model = wrapper.get_model(model_checkpoint)
    return model

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
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

def run_inference(model: torch.nn.Module, audio_path: str, demography_info: float, config: Dict[str, Any]) -> Tuple[int, str]:
    """
    Run inference to get predicted_label and transcription.

    Args:
        model (torch.nn.Module): Loaded model.
        audio_path (str): Path to the audio file.
        demography_info (float): Demographic information.
        config (Dict[str, Any]): Model configuration.

    Returns:
        Tuple[int, str]: Predicted label and transcription.
    """
    demography_tensor = torch.tensor(demography_info, dtype=torch.float16).reshape(1, 1)
    predicted_label, transcription = model.inference(audio_path, demography_tensor, config)
    return predicted_label, transcription

def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()

    # Validate arguments
    validate_arguments(args)

    # Load model configuration from YAML file
    config_path = "SpeechCARE_Linguistic_Explainability_Framework/data/model_config.yaml"  # Path to your YAML configuration file
    config = Config(load_config_from_yaml(config_path))

    # Initialize and load the model
    model = initialize_model(config, args.model_checkpoint)

    # Set predicted_label and transcription
    if args.predicted_label is not None and args.transcription is not None:
        model.predicted_label = args.predicted_label
        model.transcription = args.transcription
        print("Using provided predicted_label and transcription.")
    else:
        # Run inference to get predicted_label and transcription
        model.predicted_label, model.transcription = run_inference(model, args.audio_path, args.demography_info, config)
        print("Running inference to compute predicted_label and transcription.")

    # Initialize SHAP explainer (if needed)
    shap = LinguisticShap(model)
    html_result = shap.get_text_shap_results(args.save_path)

if __name__ == "__main__":
    main()