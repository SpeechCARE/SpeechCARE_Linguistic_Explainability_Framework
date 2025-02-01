import os
import pandas as pd

import utils as shapUtils

from SpeechCARE_Linguistic_Explainability_Framework.trainer.Trainer import Trainer 
from SpeechCARE_Linguistic_Explainability_Framework.config import Config
from SpeechCARE_Linguistic_Explainability_Framework.models.Model import TextOnlyModel_withoutGate
from SpeechCARE_Linguistic_Explainability_Framework.dataset import utils as dsUtils

from SpeechCARE_Linguistic_Explainability_Framework.utils.Utils import report


import shap

import torch



def compute_shap_values(tokenizer,input):

    text_explainer = shap.Explainer(utils.text_predict, tokenizer, output_names=labels, hierarchical_values=True)
    control_textData_shap_values = text_explainer(input)

def get_sample_df(file_path):
    """Loads and processes the samples dataset."""
    samples_df = pd.read_excel(file_path)
    return samples_df

def get_config():
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

    return config


def main():
    """Main function to execute the script."""
    global PRETRAINED_MODEL_DIR, SAMPLES_DIR, RESULTS_DIR,SIMPLE_ATTENTION, device,labels
    
    PRETRAINED_MODEL_DIR = ''
    SAMPLES_DIR = ''
    RESULTS_DIR = ''
    SIMPLE_ATTENTION = 16
    TRAIN_SUBJECTS = ''
    VALID_SUBJECTS = ''
    TEST_SUBJECTS = ''
    TRAIN_AUDIOS = ''
    VALID_AUDIOS = ''
    TEST_AUDIOS = ''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = ['control', 'mci', 'adrd']

    config = get_config()
    checkpoint = torch.load('/workspace/Final_model_age_category_128_embedding/Model/model_checkpoint.pth')

    dataset = dsUtils.get_dataset(config,TRAIN_SUBJECTS,VALID_SUBJECTS,TEST_SUBJECTS,TRAIN_AUDIOS,VALID_AUDIOS,TEST_AUDIOS)

    trainer = Trainer(config,dataset,checkpoint)

    text_explainer = shap.Explainer(shapUtils.text_predict, trainer.tokenizer, output_names=labels, hierarchical_values=True)

    samples_df = get_sample_df(os.path.join(SAMPLES_DIR, 'Explainability_Samples.xlsx')) #list
    samples_txt = samples_df['transcriptions'].to_list()
    control_textData_shap_values = text_explainer(samples_txt)

    shap_data = []
    for i, (id, explanation) in enumerate(zip(samples_df['uids'].to_list(), control_textData_shap_values)):
        data = {
            "index": id,
            "tokens": explanation.data.tolist(),  # The tokens from the text
            "shap_values": explanation.values.tolist(),  # SHAP values for each token
            "base_values": explanation.base_values.tolist(),
            "original_text": samples_txt[i],  # The original text (optional),
            "class": 'control'
        }
        shap_data.append(data)

    shapUtils.text(control_textData_shap_values)
    report("Processing complete.", True)


    # text_model = TextOnlyModel_withoutGate(trainer.model.txt_transformer, trainer.model.txt_head, trainer.model.txt_classifier)


   

if __name__ == "__main__":
    main()




