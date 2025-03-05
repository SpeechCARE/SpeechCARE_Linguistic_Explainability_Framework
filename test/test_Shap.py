import sys
import argparse
import torch
from typing import Optional, Tuple

# Add custom paths to sys.path (if needed)
sys.path.append("")

from SpeechCARE_Linguistic_Explainability_Framework.SHAP.Shap import LinguisticShap
from SpeechCARE_Linguistic_Explainability_Framework.models.ModelWrapper import ModelWrapper
from SpeechCARE_Linguistic_Explainability_Framework.Config import Config

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

def initialize_model(config: Config, model_checkpoint: str) -> torch.nn.Module:
    """
    Initialize the model wrapper and load the pretrained model.

    Args:
        config (Config): Configuration object for the model.
        model_checkpoint (str): Path to the model checkpoint.

    Returns:
        torch.nn.Module: Loaded model.
    """
    wrapper = ModelWrapper(config)
    model = wrapper.get_model(model_checkpoint)
    return model

def get_model_config() -> Config:
    """
    Create and configure the model configuration.

    Returns:
        Config: Configured model configuration.
    """
    config = Config()
    config.seed = 133
    config.bs = 4
    config.epochs = 14
    config.lr = 1e-6
    config.hidden_size = 128
    config.wd = 1e-3
    config.integration = 16  # SIMPLE_ATTENTION
    config.num_labels = 3
    config.txt_transformer_chp = config.MGTEBASE
    config.speech_transformer_chp = config.mHuBERT
    config.segment_size = 5
    config.active_layers = 12
    config.demography = 'age_bin'
    config.demography_hidden_size = 128
    config.max_num_segments = 7
    return config

def run_inference(model: torch.nn.Module, audio_path: str, demography_info: float, config: Config) -> Tuple[int, str]:
    """
    Run inference to get predicted_label and transcription.

    Args:
        model (torch.nn.Module): Loaded model.
        audio_path (str): Path to the audio file.
        demography_info (float): Demographic information.
        config (Config): Model configuration.

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

    # Initialize model configuration
    config = get_model_config()

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