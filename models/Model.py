# Pytorch
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Huggingface
from transformers import AutoModel

import SpeechCARE_Linguistic_Explainability_Framework.models.ModelWrapper as ModelWrapper 
import SpeechCARE_Linguistic_Explainability_Framework.utils.Utils as set_seed 

class TextOnlyModel_withoutGate(nn.Module):
    def __init__(self, txt_transformer, txt_head, txt_classifier):
        super(TextOnlyModel_withoutGate, self).__init__()
        self.txt_transformer = txt_transformer
        self.txt_head = txt_head
        self.txt_classifier = txt_classifier

    def forward(self, input_ids, attention_mask):
        txt_embeddings = self.txt_transformer(input_ids=input_ids, attention_mask=attention_mask)
        txt_cls = txt_embeddings.last_hidden_state[:, 0, :]
        txt_head = self.txt_head(txt_cls)
        txt_out = self.txt_classifier(txt_head)
        return txt_out
    
class TBNet_(nn.Module):
    
    def __init__ (self, config):
        super(TBNet_, self).__init__()
        ModelWrapper.set_seed(config.seed)

        self.transformer = AutoModel.from_pretrained(config.transformer_chp, trust_remote_code=True)
        embedding_dim = self.transformer.config.hidden_size

        self.projector = nn.Linear(embedding_dim, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)



    def forward (self,input,input_type="token_id", attention_mask=None):
        if attention_mask == None:
          attention_mask = torch.ones_like(input)

        if input_type == "token_id":
          output = self.transformer(input_ids = input ,
                                    attention_mask = attention_mask)
        elif input_type == "token_embedding":
          output = self.transformer(inputs_embeds = input ,
                                    attention_mask = attention_mask)
        else:
          raise ValueError('Input not specified!')



        cls = output.last_hidden_state[:, 0, :]
        x = self.projector(cls)
        x = F.tanh(x)
        x = self.classifier(x)
        # print(x.shape)

        return x
    

class MultiHeadAttentionAddNorm(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):

        super(MultiHeadAttentionAddNorm, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        # Multi-Head Attention
        attn_output, _ = self.mha(x, x, x)  # Self-attention: Q = K = V = x

        # Add & Norm
        x = self.norm(x + self.dropout(attn_output))
        return x


class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 3)  # Output 3 weights for speech, text, and demography
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        gate_weights = self.fc(x)
        return self.softmax(gate_weights)  # Ensure weights sum to 1



class TBNet(nn.Module):

    def __init__ (self, config):
        super(TBNet, self).__init__()
        set_seed(config.seed)

        self.speech_transformer = AutoModel.from_pretrained(config.speech_transformer_chp)
        self.txt_transformer = AutoModel.from_pretrained(config.txt_transformer_chp, trust_remote_code=True)

        speech_embedding_dim = self.speech_transformer.config.hidden_size
        txt_embedding_dim = self.txt_transformer.config.hidden_size

        # Trainable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, speech_embedding_dim))

        # Positional encoding
        max_seq_length = int(config.max_num_segments * ((config.segment_size / 0.02) - 1)) + 1 # +1 for CLS embedding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, speech_embedding_dim))

        num_layers = 2
        self.layers = nn.ModuleList([
            MultiHeadAttentionAddNorm(speech_embedding_dim, 4, 0.1)
            for _ in range(num_layers)
        ])

        self.speech_head = nn.Sequential(
            nn.Linear(speech_embedding_dim, config.hidden_size),
            nn.Tanh(),
        )

        self.txt_head = nn.Sequential(
            nn.Linear(txt_embedding_dim, config.hidden_size),
            nn.Tanh(),
        )

        # demography embedding head
        self.demography_head = nn.Sequential(
            nn.Linear(1, config.demography_hidden_size),  # Input is a single scalar
            nn.Tanh(),
        )

        self.speech_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.txt_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.demography_classifier = nn.Linear(config.demography_hidden_size, config.num_labels)

        # Updated Gating Network for 3 modalities
        self.weight_gate = GatingNetwork((config.hidden_size * 2) + config.demography_hidden_size)


    def forward (self, input_values, input_ids, demography, attention_mask):

        # Step 1: Original input shape: [batch_size, num_segments, seq_length]
        batch_size, num_segments, seq_length = input_values.size()

        # Step 2: Reshape to [batch_size * num_segments, seq_length]
        input_values = input_values.view(batch_size * num_segments, seq_length)

        # Step 3: Pass through transformer
        speech_embeddings = self.speech_transformer(input_values).last_hidden_state
        # Current shape after transformer: [batch_size * num_segments, num_embeddings, dim]
        txt_embeddings = self.txt_transformer(input_ids = input_ids, attention_mask = attention_mask)

        # Step 4: Reshape to [batch_size, num_segments, num_embeddings, dim]
        speech_embeddings = speech_embeddings.view(batch_size, num_segments, -1, speech_embeddings.size(-1))

        # Step 5: Flatten num_segments and num_embeddings to [batch_size, num_segments * num_embeddings, dim]
        speech_embeddings = speech_embeddings.view(batch_size, num_segments * speech_embeddings.size(2), -1)

        # Step 6: Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embedding_dim)

        # Step 7: Prepend CLS token to embeddings
        speech_embeddings = torch.cat((cls_tokens, speech_embeddings), dim=1)  # Shape: (batch_size, seq_len+1, embedding_dim)

        # Step 8: Add positional encodings
        speech_embeddings += self.positional_encoding[:, :speech_embeddings.size(1), :]  # Match the sequence length

        # Step 9: Pass through MultiHead Attention
        for layer in self.layers:
            speech_embeddings = layer(speech_embeddings)

        # Step 10: Get CLS embedding vector
        speech_cls = speech_embeddings[:, 0, :]
        txt_cls = txt_embeddings.last_hidden_state[:, 0, :]

        # demography modality processing
        demography = demography.unsqueeze(1)  # Ensure demography has shape [batch_size, 1]
        demography_x = self.demography_head(demography)

        speech_x = self.speech_head(speech_cls)
        txt_x = self.txt_head(txt_cls)

        gate_weights = self.weight_gate(torch.cat([speech_x, txt_x, demography_x], dim=1))

        speech_out = self.speech_classifier(speech_x)
        txt_out = self.txt_classifier(txt_x)
        demography_out = self.demography_classifier(demography_x)

         # Gated fusion of outputs
        weight_speech, weight_txt, weight_demography = gate_weights[:, 0], gate_weights[:, 1], gate_weights[:, 2]
        fused_output = (
            weight_speech.unsqueeze(1) * speech_out +
            weight_txt.unsqueeze(1) * txt_out +
            weight_demography.unsqueeze(1) * demography_out
        )

        return fused_output
