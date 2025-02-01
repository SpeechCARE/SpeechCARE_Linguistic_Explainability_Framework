import os
import pandas as pd

import torch


from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients

import utils as IGUtils

from SpeechCARE_Linguistic_Explainability_Framework.models.ModelWrapper import ModelWrapper
from SpeechCARE_Linguistic_Explainability_Framework.config import Config
from SpeechCARE_Linguistic_Explainability_Framework.utils.Utils import free_gpu_memory,report


def process_samples(file_path):
    """Loads and processes the samples dataset."""
    samples_df = pd.read_excel(file_path)
    samples_df[['token ids', 'reference id', 'attention_mask', 'tokens']] = samples_df.apply(
        lambda row: pd.Series(IGUtils.calculate_token_pairs(row,ref_token_id)), axis=1
    )
    samples_df[['C', 'MCI', 'ADRD', 'predicted label']] = samples_df.apply(
        lambda row: pd.Series(IGUtils.predict_label_confidence(row)), axis=1
    )
    return samples_df

def compute_attributions(model, samples_df):
    """Computes attributions using Layer Integrated Gradients."""
    lig = LayerIntegratedGradients(IGUtils.forward_func, model.transformer.embeddings.word_embeddings)
    free_gpu_memory()
    
    attributions_array, attributions_sum_array, delta_array = [], [], []
    for _, row in samples_df.iterrows():
        attributions, delta = lig.attribute(
            inputs=row['token ids'].to(device),
            target=row['true_label'],
            baselines=row['reference id'].to(device),
            additional_forward_args=(row['attention_mask'].to(device)),
            return_convergence_delta=True,
            n_steps=50
        )
        
        attributions_array.append(attributions.cpu())
        attributions_sum_array.append(IGUtils.summarize_attributions(attributions.cpu()))
        delta_array.append(delta.cpu())
    
    samples_df['attributions'] = pd.Series(attributions_array)
    samples_df['attributions_sum'] = pd.Series(attributions_sum_array)
    samples_df['delta'] = pd.Series(delta_array)

    return samples_df

def generate_visualizations(samples_df):
    """Generates visualizations for each sample."""
    
    visualizations = []
    for _, row in samples_df.iterrows():
        vis = viz.VisualizationDataRecord(
            torch.tensor(row['attributions_sum']),
            row[label_map[row['predicted label']]],
            label_map[row['predicted label']],
            label_map[row['true_label']],
            row['true_label'],
            row['attributions'].sum(),
            row['tokens'],
            row['delta']
        )
        visualizations.append(vis)
    
    samples_df['visualizations'] = pd.Series(visualizations)
    return samples_df

def get_config():
    config = Config()
    config.seed = 133
    config.hidden_size = 128
    config.num_labels = 3

    return config


def save_visualizations(samples_df):
    """Saves visualizations as HTML files."""
    for _, row in samples_df.iterrows():
        output_file = os.path.join(
            RESULTS_DIR,
            f"truth_{row['true_label']}_classified_{row['predicted label']}_conf_{str(row[label_map[row['true_label']]])}.html"
        )
        with open(output_file, "w") as file:
            file.write(viz.visualize_text([row['visualizations']]).data)

def main():
    """Main function to execute the script."""
    global PRETRAINED_MODEL_DIR, SAMPLES_DIR, RESULTS_DIR, device,label_map,ref_token_id
    
    PRETRAINED_MODEL_DIR = ''
    SAMPLES_DIR = ''
    RESULTS_DIR = ''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = {0: 'C', 1: 'MCI', 2: 'ADRD'}

    config = get_config()
    model_wrapper = ModelWrapper(config)

    model, tokenizer = model_wrapper.load_model_and_tokenizer(device,os.path.join(PRETRAINED_MODEL_DIR, "model.pt"))
    ref_token_id = tokenizer.pad_token_id

    samples_df = process_samples(os.path.join(SAMPLES_DIR, 'Explainability_Samples.xlsx'))
    samples_df = compute_attributions(model, samples_df)
    samples_df = generate_visualizations(samples_df,label_map)
    save_visualizations(samples_df,label_map)
    
    report("Processing complete. Visualizations saved successfully.", True)

if __name__ == "__main__":
    main()
