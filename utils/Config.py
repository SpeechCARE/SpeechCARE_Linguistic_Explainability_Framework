
class Config:
    def __init__(self, config_dict):
        # Load model checkpoints
        self.HUBERT = config_dict['model_checkpoints']['HUBERT']
        self.WAV2VEC2 = config_dict['model_checkpoints']['WAV2VEC2']
        self.mHuBERT = config_dict['model_checkpoints']['mHuBERT']
        self.MGTEBASE = config_dict['model_checkpoints']['MGTEBASE']
        self.WHISPER = config_dict['model_checkpoints']['WHISPER']

        # Load configuration parameters
        self.seed = config_dict['config']['seed']
        self.lr = config_dict['config']['lr']
        self.wd = config_dict['config']['wd']
        self.epochs = config_dict['config']['epochs']
        self.bs = config_dict['config']['bs']
        self.integration = config_dict['config']['integration']
        self.hidden_size = config_dict['config']['hidden_size']
        self.active_layers = config_dict['config']['active_layers']
        self.num_labels = config_dict['config']['num_labels']
        self.demography = config_dict['config']['demography']
        self.segment_size = config_dict['config']['segment_size']
        self.demography_hidden_size = config_dict['config']['demography_hidden_size']
        self.txt_transformer_chp = config_dict['config']['txt_transformer_chp']
        self.speech_transformer_chp = config_dict['config']['speech_transformer_chp']
        self.max_num_segments = config_dict['config']['max_num_segments']

    def get_subnet_insize(self):
        if self.transformer_checkpoint == self.HUBERT:
            return 768
        elif self.transformer_checkpoint == self.WAV2VEC2:
            return 768