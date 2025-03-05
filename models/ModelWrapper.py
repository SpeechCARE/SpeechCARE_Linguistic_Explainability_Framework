from models.Model import TBNet
from utils.Utils import report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



from transformers import AutoTokenizer
from transformers import Wav2Vec2FeatureExtractor

 
class ModelWrapper:
    def __init__(self, config):

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        report(f'{self.device} is available', True)
    

    def get_model(self, weights_path= None):
        """Returns the model instance."""
        model = TBNet(self.config).to(self.device)
        if weights_path!= None: model.load_state_dict(torch.load(weights_path))
        return model
    
    def get_voice_feature_extractor(self):
        return Wav2Vec2FeatureExtractor.from_pretrained(self.config.speech_transformer_chp)
       
    def get_scheduler(self,type,optimizer):
        if type == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience= 4, min_lr=1e-6, threshold=0.01)
        else:
            return None

    def init_optimizer(self, model):
        """Initializes and returns an optimizer."""
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if 'txt_transformer' in name:
                continue
            if 'bias' in name or 'LayerNorm' in name or 'norm' in name or 'positional_encoding' in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = optim.AdamW([
            {'params': decay, 'lr': 1e-5, 'weight_decay': 0.005},
            {'params': no_decay, 'lr': 1e-5, 'weight_decay': 0.0},
            {'params': model.txt_transformer.parameters(), 'lr': 1e-6, 'weight_decay': 0.005},
        ])

        return optimizer

    
    def load_model_and_tokenizer(self,device,weights_path= None):
        model = self.get_model(self.config , device ,weights_path)
        model.eval()
        model.zero_grad()
        tokenizer = AutoTokenizer.from_pretrained(self.config.transformer_chp)
        return model, tokenizer
    
  