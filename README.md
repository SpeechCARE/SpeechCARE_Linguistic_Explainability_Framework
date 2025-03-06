# Model Explainability for Text Inputs

This repository implements a **transformer-based classification pipeline** for analyzing and interpreting text inputs. To enhance interpretability, we provide tools to explain the model's decision-making process using methods such as **SHAP (SHapley Additive exPlanations)**. These methods allow us to identify and visualize the specific words, phrases, or tokens in the text that the model attends to most when making predictions. Whether you're working on sentiment analysis, text classification, or any other NLP task, this repository aims to make your model's decisions more transparent and interpretable.

## 🚀 Installation

First, install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuring `*.yml`

Before starting the training process, update the **`data/model_config.yml`** file with the appropriate paths and settings.

### ✅ Set Pretrained Checkpoints

Choose a pretrained acoustic transformer model by specifying its checkpoint in the configuration file. The pipeline supports various self-supervised speech models:

```yaml
# mHuBERT: Multilingual HuBERT model for robust speech representation learning
speech_transformer_chp: "utter-project/mHuBERT-147"
```

```yaml
# wav2vec 2.0: Self-supervised model trained on 960 hours of English speech
speech_transformer_chp: "facebook/wav2vec2-base-960h"
```

```yaml
# HuBERT: Hidden-unit BERT model trained on the LibriSpeech 960h dataset
speech_transformer_chp: "facebook/hubert-base-ls960"
```

### ✅ Set Training Hyperparameters

Change other training parameters or model configs like epoch, learning rate and etc.

---

## 🛠️ Usage

To use the provided explainability methods (SHAP) on an text input, you can run the `test_Shap.py` file using the following bash script. This script generates explanations for a given test sample and saves the results.

### Running the Script

Use the following command to run the `test_Shap.py` file:

```bash
!python SpeechCARE_Linguistic_Explainability_Framework/test/test_Shap.py --model_checkpoint $CHECKPOINTS_FILE \
                                                                  --transcription "$TRANSCRIPTION" \
                                                                  --predicted_label $PREDICTED_LABEL \
                                                                  --save_path $HTML_SAVE_PATH

```

### Arguments

- **`--model_checkpoint`**:  
  Path to the pretrained TBNet model weights. This file contains the trained model parameters required for inference.

- **`--transcription`**:  
  A string associated with the transcription to the sample's audio.

- **`--predicted_label`**:  
  A scalar value associated with the label predicted for the sample.

- **`--save_path`**:  
  Path to save the generated .html file with SHAP values visualized. This file highlights the parts of the text signal that the model attended to most.

## 📁 Repository Structure

```
├── data/                       # Contains necessary data
├── dataset/                    # Dataset architecture
├── models/                      # Model architecture
├── utils/                      # Utility scripts for preprocessing and evaluation
├── test/                 # A sample script for using the explanation on text data
├── results_SHAP.ipynb                      # A notebook sample to show the output of the explanation method used
├── requirements.txt              # Dependencies for the project
```

---
