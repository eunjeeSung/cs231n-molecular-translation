import torch
import torch.nn as nn

from models.attention import Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class DecoderRNN(nn.Module):
    """model adapted from https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/data
    readapted from https://www.kaggle.com/pasewark/pytorch-resnet-lstm-with-attention
    """

    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        
    def forward(self, features, captions):       
        #vectorize the caption
        embeds = self.embedding(captions)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(features) # (batch_size, decoder_dim)
        
        # get the seq length to iterate
        seq_length = len(captions[0])-1 # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)
                
        for s in range(seq_length):
            alpha, attn_weight = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], attn_weight), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha     
        
        return preds
    
    def generate_caption(self, features, max_len=200, itos=None,stoi=None):
        # Inference part
        # Given the image features generate the captions
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #starting input
        word = torch.full((batch_size,1),stoi['<sos>']).to(device)
        embeds = self.embedding(word)

        captions=torch.zeros((batch_size,202),dtype=torch.long).to(device)
        captions[:,0]=word.squeeze()
        
        for i in range(202):
            alpha, attn_weights = self.attention(features, h)
        
            lstm_input = torch.cat((embeds[:, 0], attn_weights), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)  
            
            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            #save the generated word
            captions[:,i]=predicted_word_idx

            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx).unsqueeze(1)
        
        #covert the vocab idx to words and return sentence
        return captions
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c