import copy

import PIL
from PIL import Image

import Levenshtein

import numpy as np
import nltk

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop

from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors, pad_tensor

# IO
transform = Compose([
    #RandomHorizontalFlip(),
    Resize((160,320), PIL.Image.BICUBIC),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_train_file_path(image_ids):
    # print(image_id)
    # return "../input/bms-molecular-translation/train/{}/{}/{}/{}.png".format(
    #     image_id[0], image_id[1], image_id[2], image_id 
    # )
    return [
        "./input/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id)
        for image_id in image_ids
    ]

def get_test_file_path(image_ids):
    return [
        "./input/test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id)
        for image_id in image_ids
    ]


# NLP
class CapsCollateExperiment:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self, batch):
        imgs = [img.unsqueeze(0) for img, _ in batch]
        imgs = torch.cat(imgs,dim=0)
        
        note_targets = [vecs[0] for img, vecs in batch]
        note_targets = pad_sequence(note_targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        c_targets = [vecs[1] for img, vecs in batch]
        c_targets = pad_sequence(c_targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        
        h_targets = [vecs[2] for img, vecs in batch]
        h_targets = pad_sequence(h_targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        b_targets = [vecs[3] for img, vecs in batch]
        b_targets = pad_sequence(b_targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        t_targets = [vecs[4] for img, vecs in batch]
        t_targets = pad_sequence(t_targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        m_targets = [vecs[5] for img, vecs in batch]
        m_targets = torch.stack(m_targets).unsqueeze(1)
        
        s_targets = [vecs[6] for img, vecs in batch]
        s_targets = torch.stack(s_targets).unsqueeze(1)

        i_targets = [vecs[7] for img, vecs in batch]
        i_targets = pad_sequence(i_targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        targets = (note_targets, c_targets, h_targets, b_targets, t_targets, m_targets, s_targets, i_targets)
        return imgs, targets

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        inchis = [item[1] for item in batch]
        inchis = pad_sequence(inchis, batch_first=self.batch_first, padding_value=self.pad_idx)

        return imgs, inchis, None


def string_to_ints(string, stoi):
    l=[stoi['<sos>']]
    i = 0
    while type(string) != float and i < len(string):
        if string[i:i+3] in stoi:
            l.append(stoi[ string[i:i+3] ])
            i += 3            
        elif string[i:i+2] in stoi:
            l.append(stoi[ string[i:i+2] ])
            i += 2
        else:
            l.append(stoi[string[i]])
            i += 1
    l.append(stoi['<eos>'])
    return l

def ints_to_string(l):
    return ''.join(list(map(lambda i:itos[i],l)))

def c_string_to_ints(string, stoi):
    l=[stoi['<sos>']]
    string = string.replace('-', ' ').replace(',', ' , ').replace('(', ' ( ').replace(')', ' ) ').split(' ')
    for part in string:
        if part == '':
            continue
        l.append(stoi[part])
    l.append(stoi['<eos>'])
    return l    
    
def t_string_to_ints(string, stoi):
    l=[stoi['<sos>']]
    string = string.split(',')
    for part in string:
        if part == '':
            continue
        l.append(stoi[part])
    l.append(stoi['<eos>'])
    return l        


# Transformer
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# misc.
def tensor_to_captions(ten, stoi, itos):
    l=ten.tolist()
    ret=[]
    for ls in l:
        temp=''
        #for i in ls[1:]:
        for i in ls:
            if i==stoi['<eos>'] or i==stoi['<pad>']:
                break
            temp=temp+itos[i]
        ret.append(temp)
    return ret

def tensor_to_captions_multihead(tensor, dictionaries, inverse_dictionaries):
    note_preds, c_preds, h_preds, b_preds, t_preds, m_preds, s_preds, i_preds = tensor
    general_to_i, t_to_i, m_to_i, s_to_i = dictionaries
    i_to_general, i_to_t, i_to_m, i_to_s = inverse_dictionaries

    note_caption = tensor_to_captions(note_preds, general_to_i, i_to_general)
    c_caption = tensor_to_captions(c_preds, general_to_i, i_to_general)
    h_caption = tensor_to_captions(h_preds, general_to_i, i_to_general)
    b_caption = tensor_to_captions(b_preds, general_to_i, i_to_general)
    t_caption = tensor_to_captions(t_preds, t_to_i, i_to_t)
    m_preds, s_preds = m_preds.reshape(-1), s_preds.reshape(-1)
    m_caption = [i_to_m[i] for i in m_preds.tolist()]
    s_caption = [i_to_s[i] for i in s_preds.tolist()]
    i_caption = tensor_to_captions(i_preds, general_to_i, i_to_general)

    captions = (note_caption, c_caption, h_caption, b_caption, t_caption, m_caption, s_caption, i_caption)
    return captions

def save_model(model, num_epochs, embed_size, vocab_size,
            attention_dim, encoder_dim, decoder_dim):
    model_state = {
        'num_epochs':num_epochs,
        'embed_size':embed_size,
        'vocab_size': vocab_size,
        'attention_dim':attention_dim,
        'encoder_dim':encoder_dim,
        'decoder_dim':decoder_dim,
        'state_dict':model.state_dict()        
    }
    torch.save(model_state,'checkpoints/attention_model_state.pth')

def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        # score = Levenshtein.distance(true, pred)
        score = nltk.edit_distance(pred, true)
        scores.append(score)
    avg_score = torch.mean(torch.Tensor(scores))
    return avg_score