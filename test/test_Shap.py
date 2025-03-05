import sys
import argparse
import torch

# Add custom paths to sys.path (if needed)
sys.path.append("")


from SpeechCARE_Linguistic_Explainability_Framework.SHAP.Shap import LinguisticShap
from SpeechCARE_Linguistic_Explainability_Framework.models.ModelWrapper import ModelWrapper
from SpeechCARE_Linguistic_Explainability_Framework.Config import Config

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Test Linguistic Explainability Framework")

    # Required arguments
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the PyTorch model to use as the pretrained model.")
    parser.add_argument("--demography_info", type=float, required=True,
                        help="Demographic information of shape [batch_size, 1].")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to the audio file of the selected sample.")
    
    # Optional arguments
  

    return parser.parse_args()

def initialize_model(config, model_checkpoint):
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



def main():
    # Parse command-line arguments
    args = parse_arguments()

    config = Config()
    # Initialize model configuration
    SIMPLE_ATTENTION = 16
    config = Config()
    config.seed = 133
    config.bs = 4
    config.epochs = 14
    config.lr = 1e-6
    config.hidden_size = 128
    config.wd = 1e-3
    config.integration = SIMPLE_ATTENTION
    config.num_labels = 3
    config.txt_transformer_chp = config.MGTEBASE
    config.speech_transformer_chp = config.mHuBERT
    config.segment_size = 5
    config.active_layers = 12
    config.demography = 'age_bin'
    config.demography_hidden_size = 128
    config.max_num_segments = 7
    # Initialize and load the model
    model = initialize_model(config, args.model_checkpoint)

    # Run inference to get transcription and predicted_label for the model or set them manually
    demography_tensor = torch.tensor(args.demography_info, dtype=torch.float16).reshape(1, 1) # Convert demographic information to a tensor
    predicted_label , _ = model.inference(args.audio_path, demography_tensor, config)

    print(type(predicted_label))
    print((predicted_label.shape))
    transcription = model.transcription #save
    print(type(transcription))


    # Initialize SHAP explainer
    # shap = LinguisticShap(model)
    # html_result = shap.get_text_shap_results()



if __name__ == "__main__":
    main()