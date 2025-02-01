import random
import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from data.Dataset import AudioDataset

from collections import defaultdict

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, auc, roc_auc_score

import matplotlib.pyplot as plt

import pandas as pd

def report(text, space = False):
    print(text)
    if space: print('-' * 50)

def free_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Utils():

    def __init__(self, config):
        set_seed(config.seed)
        self.config = config
        self.performance = defaultdict(lambda: 0)



    def get_wights(self, data):
        labels = data.label.values
        classes = np.unique(labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return class_weights


    def get_dataloaders (self, dataset, batch_size, tokenizer, feature_extractor):

        g = torch.Generator()
        g.manual_seed(self.config.seed)

        cf = lambda batch: self.collate_fn(batch, tokenizer=tokenizer, feature_extractor=feature_extractor)

        train_dl = DataLoader(AudioDataset(dataset['TRAIN']), batch_size = batch_size, collate_fn = cf, shuffle=True, generator=g, pin_memory=True)
        val_dl = DataLoader(AudioDataset(dataset['VALID']), batch_size = 2, collate_fn = cf, shuffle=False, generator=g)
        test_dl = DataLoader(AudioDataset(dataset['TEST']), batch_size = 2, collate_fn = cf, shuffle=False, generator=g)

        return train_dl, val_dl, test_dl



    def collate_fn(self, batch, tokenizer, feature_extractor):

        waveforms_list = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        demography = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        uids = [item[4] for item in batch]

        #Preprocessing textual data
        encoded_batch = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_batch.input_ids
        attention_masks = encoded_batch.attention_mask

        # Preprocessing audio data
        # Step 1: Ensure we have the fixed number of segments
        for i in range(len(waveforms_list)):
            waveforms = waveforms_list[i]
            current_segments = len(waveforms)

            if current_segments < self.config.max_num_segments:
                padding_needed = self.config.max_num_segments - current_segments
                waveforms_list[i].extend([torch.zeros_like(waveforms[0])] * padding_needed)


        # Step 2: Flatten all segments across all uids into one tensor
        all_segments = [waveform.squeeze().tolist() for waveforms in waveforms_list for waveform in waveforms]

        # Step 3: Process all segments at once through the feature extractor, using padding
        features = feature_extractor(all_segments, sampling_rate=16000, return_tensors="pt", padding=True)
        padded_inputs_values = features.input_values.squeeze()

        # Step 4: Reorganize features to match the desired shape [batch_size, num_segments, max_time_steps]
        batch_inputs_values = []
        start_idx = 0
        for waveforms in waveforms_list:
            num_segments = len(waveforms)
            end_idx = start_idx + num_segments
            batch_inputs_values.append(padded_inputs_values[start_idx:end_idx])
            start_idx = end_idx

        labels = torch.tensor(labels, dtype=torch.long) if labels[0] is not None else None

        return torch.stack(batch_inputs_values), input_ids, torch.tensor(demography, dtype=torch.float32), attention_masks, labels, uids



    def update_lr(self, optimizer, scheduler, loss, epoch):
        pre_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss)
        pos_lr = optimizer.param_groups[0]['lr']
        if pre_lr != pos_lr: print(f' -> Learning rate reduced at epoch {epoch + 1}!')



    def save_model(self, model, path, epoch):
        torch.save(model.state_dict(), path)
        self.performance['best epoch'] = epoch



    def save_checkpoint(self, model, optimizer, scheduler, epoch, minloss, loss_list, metric_list, save_path):
        checkpoint = {
            'minloss': minloss,
            'loss_list': loss_list,
            'metric_list' : metric_list,
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_epock': self.performance['best epoch']
            }
        torch.save(checkpoint, save_path)



    def load_checkpoint(self, checkpoint, model, optimizer, scheduler):
        minloss = checkpoint['minloss']
        loss_list = checkpoint['loss_list']
        metric_list = checkpoint['metric_list']
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        self.performance['best epoch'] = checkpoint['best_epock']

        return model, optimizer, scheduler, epoch, minloss, loss_list ,metric_list

    def save_training(self, loss_list, metric_list, path):

        training_df = pd.DataFrame({'transformer_model': [f'{self.config.speech_transformer_chp} / {self.config.txt_transformer_chp}'] * len(loss_list),
                                'active layers': [self.config.active_layers] * len(loss_list),
                                'hidden size': [self.config.hidden_size] * len(loss_list),
                                'epoch': list(range(1, len(loss_list) + 1)),
                                'train loss': [item[0] for item in loss_list],
                                'validation loss': [item[1] for item in loss_list],
                                'train metric': [item[0] for item in metric_list],
                                'valid metric': [item[1] for item in metric_list],
                                })
        training_df.to_excel(path, index=False)
        return training_df


    def save_result(self, loss, pred_probs, pred_labels, true_labels):
        self.performance['loss'] = loss
        self.performance['auc'] = roc_auc_score(true_labels, pred_probs, multi_class='ovr')  # or 'ovo' for one-vs-one
        self.performance['prec'] = precision_score(true_labels, pred_labels, average='weighted')
        self.performance['recall'] = recall_score(true_labels, pred_labels, average='weighted')
        self.performance['f1'] = f1_score(true_labels, pred_labels, average='weighted')
        self.performance['acc'] = accuracy_score(true_labels, pred_labels)

    def save_probs(self,pred_probs, pred_y, true_y, ids, path, valid_df):

        prob_df = pd.DataFrame({'transformer_model': f'{self.config.speech_transformer_chp} / {self.config.txt_transformer_chp}',
                                'active layers': [self.config.active_layers] * len(ids),
                                'hidden size': [self.config.hidden_size] * len(ids),
                                'uid':ids,
                                'label': true_y,
                                'predicted label': pred_y,
                                'C':pred_probs[:, 0],
                                'MCI':pred_probs[:, 1],
                                'ADRD':pred_probs[:, 2]
                                })

        compelete_probs_df = prob_df.merge(valid_df[['uid', 'language', 'task']], on='uid', how='left')

        compelete_probs_df.to_excel(path, index=False)
        return prob_df

    
    def save_result(self, performance, path):
        result_df = pd.DataFrame({'transformer model': f'{self.config.speech_transformer_chp} / {self.config.txt_transformer_chp}',
                                'active layers': self.config.active_layers,
                                'hidden size': self.config.hidden_size,
                                'seed': self.config.seed,
                                'batch size': self.config.bs,
                                'epochs': self.config.epochs,
                                'learning rate': self.config.lr,
                                'best epoch' : performance['best epoch'],
                                'f1': performance['f1'],
                                'precision': performance['prec'],
                                'recall': performance['recall'],
                                'AUC': performance['auc'],
                                'accuracy': performance['acc'],
                                }, index = [0])

        result_df.to_excel(path, index = False)
        return result_df

    def plot_training (self, loss_list, metric_list, labels, title= ''):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5))
        fig.subplots_adjust(wspace = .2)
        self.plotLoss(ax1, loss_list, labels)
        self.plotMetric(ax2, metric_list, labels)
        plt.show()



    def plotLoss (self, ax, loss_list, labels):
        for i, label in enumerate(labels):
          ax.plot(loss_list[:, i], label = label)
        ax.set_title("Loss Curvess", fontsize = 9)
        ax.set_ylabel("Loss", fontsize = 8)
        ax.set_xlabel("Epoch", fontsize = 8)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
        ax.legend(prop = {'size': 8})
        ax.grid()



    def plotMetric (self, ax, metric_list, labels):
        for i, label in enumerate(labels):
          ax.plot(metric_list[:, i], label = label)
        ax.set_title("Metric Curves", fontsize = 9)
        ax.set_ylabel("Score", fontsize = 8)
        ax.set_xlabel("Epoch", fontsize = 8)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 8)
        ax.legend(prop = {'size': 8})
        ax.grid()


