class Config():

    HUBERT = 'facebook/hubert-base-ls960'
    WAV2VEC2 = 'facebook/wav2vec2-base-960h'
    mHuBERT = 'utter-project/mHuBERT-147'
    MGTEBASE = 'Alibaba-NLP/gte-multilingual-base'
    WHISPER = "openai/whisper-large-v3-turbo"

    def __init__(self, seed= None, lr= None, wd= None,
                 epochs= None, bs= None,
                 integration = None, hidden_size= None, dropout= None,
                 active_layers = 12, num_labels=2,demography = 'age_bin',segment_size = 5,demography_hidden_size = 128):
        self.seed = seed
        self.bs = bs
        self.epochs = epochs
        self.lr = lr
        self.hidden_size = hidden_size
        self.wd = wd
        self.integration = integration
        self.num_labels = num_labels
        self.txt_transformer_chp = self.MGTEBASE
        self.speech_transformer_chp = self.mHuBERT
        self.segment_size = segment_size
        self.active_layers = active_layers
        self.demography = demography
        self.demography_hidden_size = demography_hidden_size
        self.dropout = dropout

    def get_subnet_insize(self):
        if self.transformer_checkpoint == self.HUBERT:
            return 768
        elif self.transformer_checkpoint == self.WAV2VEC2:
            return 768
        
