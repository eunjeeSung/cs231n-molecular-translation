import PIL
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop


# IO
transform = Compose([
    #RandomHorizontalFlip(),
    Resize((256,256), PIL.Image.BICUBIC),
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

def get_test_file_path(image_id):
    return "./input/test/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id 
    )


# NLP
class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets


def string_to_ints(string, stoi):
    l=[stoi['<sos>']]
    for s in string:
        l.append(stoi[s])
    l.append(stoi['<eos>'])
    return l
    
def ints_to_string(l):
    return ''.join(list(map(lambda i:itos[i],l)))


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
    torch.save(model_state,'attention_model_state.pth')