
import torch
from torch.utils.data import Dataset



class AudioDataset(Dataset):

    def __init__(self, config, data_df,TRANSCRIP, DEMOGRAPHY, SEGMENTS):
        # print(data_df)
        self.audio_data = data_df[SEGMENTS]
        self.text_data = data_df[TRANSCRIP]
        self.participants = self.audio_data['uid'].unique()
        self.demography = data_df[DEMOGRAPHY]
        self.config = config


    def __len__(self):
        return len(self.participants)


    def __getitem__(self, item):

        uid = self.participants[item]
        text = self.text_data[self.text_data['uid'] == uid]['transcription'].values[0]
        segments = self.audio_data[self.audio_data['uid'] == uid].sort_values(by="segment")
        label = segments['label'].values[0] if 'label' in segments.columns else None
        demography =  self.demography[self.demography['uid'] == uid][self.config.demography].values[0]

        waveforms = []
        for _, row in segments.iterrows():
            tensor_path = row['path']
            waveform = torch.load(tensor_path)
            waveforms.append(waveform)

        return waveforms, text, demography, label, uid
