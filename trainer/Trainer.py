from SpeechCARE_Linguistic_Explainability_Framework.utils import Utils,set_seed, report

import copy

import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch import nn

from SpeechCARE_Linguistic_Explainability_Framework.models.ModelWrapper import ModelWrapper

from transformers import AutoTokenizer
from transformers import Wav2Vec2FeatureExtractor

from sklearn.metrics import classification_report, f1_score


class Trainer(ModelWrapper):

    def __init__ (self, config, dataset, checkpoint):
            super(Trainer, self).__init__()
            self.dataset = dataset
            self.utils = Utils(config)
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            report(f'{self.device} is available', True)

            self.model, self.tokenizer = self.load_model_and_tokenizer(self.device) 
            report('Tokenizer & Model loaded successfully', True)
            feature_extractor = self.get_voice_feature_extractor()
            report('FeatureExtractor loaded successfully', True)

            self.optimizer = self.init_optimizer(self.model)
            self.scheduler = self.get_scheduler(self,"ReduceLROnPlateau",self.optimizer) 

            if checkpoint == None:
                self.epoch, self.minloss= 0, float('inf')
                self.loss_list, self.metric_list = [], []
                report('Model loaded successfully', True)
            else:
                (self.model, self.optimizer, self.scheduler, self.epoch,
                 self.minloss, self.loss_list, self.metric_list) = self.utils.load_checkpoint(checkpoint,
                                                                                                self.model,
                                                                                                self.optimizer,
                                                                                                self.scheduler
                                                                                                )
                self.model = self.model.to(self.device)
                report('Checkpoint loaded successfully', True)

            self.train_dataloader, self.val_dataloader, self.test_dataloader = self.utils.get_dataloaders(dataset,
                                                                                                          config.bs,
                                                                                                          self.tokenizer,
                                                                                                          feature_extractor)
            report(f'Number of train batches: {len(self.train_dataloader)}, (bs={config.bs})')
            report(f'Number of validation batches: {len(self.val_dataloader)}, (bs={2})')
            report(f'Number of test batches: {len(self.test_dataloader)}, (bs={2})', True)



    def train(self, model, dataloader, criterion, epoch):

            model.train()
            pred_y, true_y, loss_list = [], [], []

            for step, batch in enumerate(dataloader):
           
                (input_values, input_ids, demography, attention_masks, labels) = [t.to(self.device) for t in batch[:5]]
      
                model.zero_grad()
                outputs = model(input_values, input_ids, demography, attention_masks)

                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())
                pred_y.extend(torch.argmax(outputs, dim = 1).cpu().numpy())
                true_y.extend(labels.cpu().numpy())

                if step % 10 == 0:
                    print(f'\r[{epoch + 1}][{step}] -> Loss = {np.mean(loss_list):.4f}', end='')

            return np.mean(loss_list) , pred_y, true_y



    def evaluate(self, model, dataloader, criterion):

        model.eval()
        pred_probs, pred_y, true_y, loss_list, uid_list = [], [], [], [], []

        for step, batch in enumerate(dataloader):
            (input_values, input_ids, demography, attention_masks, labels) = [t.to(self.device) for t in batch[:5]]
            uids = batch[5]

            with torch.no_grad():
                outputs = model(input_values, input_ids, demography, attention_masks)
                loss = criterion(outputs, labels)

            output_probs = F.softmax(outputs, dim = 1)
            loss_list.append(loss.item())


            loss_list.append(loss.item())
            pred_probs.extend(output_probs.detach().cpu().tolist())
            pred_y.extend(torch.argmax(output_probs, dim = 1).cpu().numpy())
            true_y.extend(labels.cpu().numpy())
            uid_list.extend(uids)

        return np.mean(loss_list), pred_y, true_y, pred_probs, uid_list


    
    def train_and_evaluate(self, n_epochs):
        print('')

        # wights = self.utils.get_wights(self.dataset[TRAIN]).to(self.device)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epoch, n_epochs):

            self.train_dataloader.generator.manual_seed(self.config.seed + epoch)

            train_loss, train_pred_y, train_true_y = self.train(self.model, self.train_dataloader, criterion, epoch)
            val_loss, val_pred_y, val_true_y, _, _ = self.evaluate(self.model, self.val_dataloader, criterion)

            self.utils.update_lr(self.optimizer, self.scheduler, train_loss, epoch)

            t_f1 = f1_score(train_true_y, train_pred_y, average='weighted')
            v_f1 = f1_score(val_true_y, val_pred_y, average='weighted')

            self.loss_list.append([train_loss, val_loss])
            self.metric_list.append([t_f1, v_f1])

            if val_loss <= self.minloss:
                self.minloss = val_loss
                self.utils.save_model(self.model, self.MODEL_PATH, epoch)

            self.utils.save_checkpoint(self.model, self.optimizer, self.scheduler, epoch+1,
                            self.minloss, self.loss_list, self.metric_list, self.TRAIN_CKP_PATH)

            print(f'\r[{epoch+1}]--> (Train) Loss: {train_loss:.4f}, F1 score: {t_f1*100:.1f} | (VAl) Loss: {val_loss:.4f}, F1 score: {v_f1*100:.1f}')

        self.utils.plot_training(np.array(self.loss_list), np.array(self.metric_list), ['Train', 'Validation'])

        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(torch.load(self.MODEL_PATH))
        return best_model, self.loss_list, self.metric_list

    def predict(self, model, test_df):

        model.eval()
        criterion = nn.CrossEntropyLoss()

        for step, batch in enumerate(self.test_dataloader):

            (input_values, input_ids, demography, attention_masks) = [t.to(self.device) for t in batch[:4]]
            uids = batch[5]

            with torch.no_grad():
                outputs = model(input_values, input_ids, demography, attention_masks)

            output_probs = F.softmax(outputs, dim = 1)
            pred_y = torch.argmax(output_probs, dim = 1).cpu().numpy()
            output_probs = output_probs.detach().cpu().tolist()


            for i in range(len(uids)):
                test_df.loc[test_df['uid'] == uids[i], 'C'] = output_probs[i][0]
                test_df.loc[test_df['uid'] == uids[i], 'MCI'] = output_probs[i][1]
                test_df.loc[test_df['uid'] == uids[i], 'ADRD'] = output_probs[i][2]
                test_df.loc[test_df['uid'] == uids[i], 'predicted label'] = pred_y[i]

        return test_df



    def get_result(self, model, target_classes):
        loss, pred_y, true_y,pred_probs, ids = self.evaluate(model, self.val_dataloader, nn.CrossEntropyLoss())
        self.utils.save_result(loss, pred_probs, pred_y, true_y)
        print(classification_report(true_y, pred_y, target_names= target_classes))
        return self.utils.performance, pred_probs, pred_y, true_y, ids

    
    def predict_with_weights(self, model, test_df):
        
        model.eval()
        criterion = nn.CrossEntropyLoss()

        for step, batch in enumerate(self.test_dataloader):

            (input_values, input_ids, demography, attention_masks) = [t.to(self.device) for t in batch[:4]]
            uids = batch[5]

            with torch.no_grad():
                fused_output, speech_out, text_out, demography_out, weight_speech, weight_txt, weight_demography = model(input_values, input_ids, demography, attention_masks)
                
            output_probs = F.softmax(fused_output, dim = 1).detach().cpu().tolist()
            speech_out = speech_out.detach().cpu()
            text_out = text_out.detach().cpu()
            weight_speech = weight_speech.detach().cpu()
            weight_txt = weight_txt.detach().cpu()

            for i in range(len(uids)):
                test_df[uids[i]]['C'] = output_probs[i][0]
                test_df[uids[i]]['MCI'] = output_probs[i][1]
                test_df[uids[i]]['ADRD'] = output_probs[i][2]
                test_df[uids[i]]['speechScore'] = speech_out[i].tolist()
                test_df[uids[i]]['textScore'] = text_out[i].tolist()
                test_df[uids[i]]['demogScore'] = demography_out[i].tolist()
                test_df[uids[i]]['speechWeight'] = weight_speech[i].tolist()
                test_df[uids[i]]['textWeight'] = weight_txt[i].tolist()
                test_df[uids[i]]['demogWeight'] = weight_demography[i].tolist()


        return test_df