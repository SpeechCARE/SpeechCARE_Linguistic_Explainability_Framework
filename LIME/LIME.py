import torch

import pandas as pd

import utils as limeUtils
from lime.lime_text import LimeTextExplainer

from SpeechCARE_Linguistic_Explainability_Framework.utils.Utils import free_gpu_memory, report
from SpeechCARE_Linguistic_Explainability_Framework.models.ModelWrapper import ModelWrapper
from SpeechCARE_Linguistic_Explainability_Framework.config import Config

import os

def process_samples(file_path):
    """Loads and processes the samples dataset."""
    samples_df = pd.read_excel(file_path)
    return samples_df

def save_visualization(samples_df):
    """Computes attributions using Layer Lime."""
    explainer = LimeTextExplainer(class_names, verbose=True)

    for index, row in samples_df.iterrows():
        output_file = os.path.join('results' ,f"truth_{row['true_label']}_classified_{row['predicted label']}_conf_{str(row[label_map[row['true_label']]])}_LIME.html")
        sample_text = row['transcription']
        explanation = explainer.explain_instance(sample_text, limeUtils.model_predict_lime, num_features=10,num_samples=500,labels=(row['true_label']) )
        free_gpu_memory()
        with open(output_file, "w") as file:
            file.write(explanation.as_html(labels=(row['true_label'])))

  
    return samples_df

def get_config():
    config = Config()
    config.seed = 133
    config.hidden_size = 128
    config.num_labels = 3

    return config

def main():
    """Main function to execute the script."""
    global PRETRAINED_MODEL_DIR, SAMPLES_DIR, RESULTS_DIR, device,class_names,label_map,model,tokenizer
    
    PRETRAINED_MODEL_DIR = ''
    SAMPLES_DIR = ''
    RESULTS_DIR = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names=['Control', 'MCI', 'ADRD']
    label_map = {0: 'C', 1: 'MCI', 2: 'ADRD'}


    config = get_config()
    model_wrapper = ModelWrapper(config)

    model, tokenizer = model_wrapper.load_model_and_tokenizer(device,os.path.join(PRETRAINED_MODEL_DIR, "model.pt"))

    samples_df = process_samples(os.path.join(SAMPLES_DIR, 'Explainability_Samples.xlsx'))
    save_visualization(samples_df)
    
    report("Processing complete. Visualizations saved successfully.", True)

if __name__ == "__main__":
    main()
