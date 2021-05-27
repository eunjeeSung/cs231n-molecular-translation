import torch
import torch.nn as nn

from .attention import Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class DecoderRNN(nn.Module):
    """model adapted from https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/data
    readapted from https://www.kaggle.com/pasewark/pytorch-resnet-lstm-with-attention
    """

    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()  
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_size = embed_size
        self.drop_prob = drop_prob
        self.SEQUENTIAL_LAYERS = ['note', 'c', 'h', 'b', 't', 'i']
        self.CATEGORICAL_LAYERS = ['m', 's']

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim) 

        # For sequential layers
        self.embeddings = self._build_embedding_layers(self.SEQUENTIAL_LAYERS)
        self.lstm_cells = self._build_lstm_layers(self.SEQUENTIAL_LAYERS)
        self.fcns = self._build_sequential_fcn(self.SEQUENTIAL_LAYERS)
        self.dropouts = self._build_sequential_dropout(self.SEQUENTIAL_LAYERS)
        self.attentions = self._build_sequential_attention(self.SEQUENTIAL_LAYERS)

        # For categorical layers
        self.fcn_m = nn.Linear(decoder_dim, 5)
        self.fcn_s = nn.Linear(decoder_dim, 2)
        self.drop_m = nn.Dropout(drop_prob)
        self.drop_s = nn.Dropout(drop_prob)
        
    def forward(self, features, vecs):       
        self.sequential_vecs = [vecs[i].to(device) for i in [0, 1, 2, 3, 4, 7]]

        self.batch_size = vecs[1].size(0)
        num_features = features.size(1)
        
        # Initialize LSTM state        
        h, c = self.init_hidden_state(features) # (batch_size, decoder_dim)
        
        # Sequential layers
        sequential_preds = self._forward_sequentials(features, self.sequential_vecs, h, c)
        note_preds, c_preds, h_preds, b_preds, t_preds, i_preds = sequential_preds

        # Categorical layers
        m_preds = self.fcn_m(self.drop_m(h))
        s_preds = self.fcn_s(self.drop_s(h))
        return (note_preds, c_preds, h_preds, b_preds, t_preds, m_preds, s_preds, i_preds)

    def generate_caption(self, features, dictionaries=None, inverse_dictionaries=None, max_len=200):
        # Inference part
        # Given the image features generate the captions
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        # Dictionaries
        general_to_i, t_to_i, m_to_i, s_to_i = dictionaries
        
        # Sequential layers
        sequential_preds = self._get_sequential_preds(features, h, c, general_to_i, batch_size)
        note_preds, c_preds, h_preds, b_preds, t_preds, i_preds = sequential_preds

        # Categorical layers        
        m_preds = self.fcn_m(self.drop_m(h)).argmax(dim=1)
        s_preds = self.fcn_s(self.drop_s(h)).argmax(dim=1)

        #covert the vocab idx to words and return sentence
        return (note_preds, c_preds, h_preds, b_preds, t_preds, m_preds, s_preds, i_preds)

    def _build_embedding_layers(self, target_layers):
        layers = [nn.Embedding(self.vocab_size, self.embed_size) for l in target_layers]
        return nn.ModuleList(layers)

    def _build_lstm_layers(self, target_layers):
        """Returns LSTM layers for note, c, h, b, t, and i.
        """
        layers = [nn.LSTMCell(self.embed_size+self.encoder_dim, self.decoder_dim, bias=True) \
                for l in target_layers]
        return nn.ModuleList(layers)

    def _build_sequential_fcn(self, target_layers):
        layers = [nn.Linear(self.decoder_dim, self.vocab_size) for l in target_layers]
        return nn.ModuleList(layers)

    def _build_sequential_dropout(self, target_layers):
        layers = [nn.Dropout(self.drop_prob) for l in target_layers]
        return nn.ModuleList(layers)

    def _build_sequential_attention(self, target_layers):
        layers = [Attention(self.encoder_dim, self.decoder_dim, self.attention_dim) for l in target_layers]
        return nn.ModuleList(layers)

    def _forward_sequentials(self, features, sequential_vecs, h, c):
        return (self._forward_sequential(features, vec, i, h, c) \
                    for i, vec in enumerate(sequential_vecs))

    def _forward_sequential(self, features, target_vec, i, h, c):
        embeds = self.embeddings[i](target_vec.to(device))
        
        # Get the seq length to iterate
        seq_length = len(target_vec[0]) - 1
        
        # Iterate and predict chracter-by-character
        preds = torch.zeros(self.batch_size, seq_length, self.vocab_size).to(device)
        for s in range(seq_length):
            alpha, attn_weight = self.attentions[i](features, h)
            lstm_input = torch.cat((embeds[:, s], attn_weight), dim=1)
            h, c = self.lstm_cells[i](lstm_input, (h, c))
            preds[:,s] = self.fcns[i](self.dropouts[i](h))
        return preds

    def _get_sequential_preds(self, features, h, c, general_to_i, batch_size):
        return (self._get_sequential_pred(features, i, h, c, general_to_i, batch_size) \
                for i, _ in enumerate(self.sequential_vecs))

    def _get_sequential_pred(self, features, i, h, c, general_to_i, batch_size):
        #starting input
        word = torch.full((batch_size, 1), general_to_i['<sos>']).to(device)
        embeds = self.embeddings[i](word)

        seq_length = 202
        preds = torch.zeros(batch_size, seq_length).to(device)
        preds[:, 0] = word.squeeze()

        for s in range(seq_length):
            alpha, attn_weights = self.attentions[i](features, h)

            lstm_input = torch.cat((embeds[:, 0], attn_weights), dim=1)
            h, c = self.lstm_cells[i](lstm_input, (h, c))

            output = self.fcns[i](self.dropouts[i](h))
            output = output.view(batch_size, -1)

            # save the generated word
            pred_idx = output.argmax(dim=1)
            preds[:, s] = pred_idx

            # send generated word as the next caption
            embeds = self.embeddings[i](pred_idx).unsqueeze(1)
        return preds
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c