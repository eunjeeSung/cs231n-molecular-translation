import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import yaml

import numpy as np
import pandas as pd
import seaborn as sns
import time
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim

from data import InputDatasetTest
from util import CapsCollate, get_train_file_path, get_test_file_path, tensor_to_captions, save_model, transform
from models.resnet_lstm import EncoderDecodertrain18


if __name__ == "__main__":
    # Get train and test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train = pd.read_csv('./input/train_labels.csv')
    print(f'train.shape: {train.shape}')


    # Make vocab
    words=set()
    for st in train['InChI']:
        words.update(set(st))
    print(f'words length: {len(words)}')

    vocab=list(words)
    vocab.append('<sos>')
    vocab.append('<eos>')
    vocab.append('<pad>')
    stoi={'C': 0,')': 1,'P': 2,'l': 3,'=': 4,'3': 5,'N': 6,'I': 7,'2': 8,'6': 9,'H': 10,'4': 11,'F': 12,'0': 13,'1': 14,'-': 15,'O': 16,'8': 17,
    ',': 18,'B': 19,'(': 20,'7': 21,'r': 22,'/': 23,'m': 24,'c': 25,'s': 26,'h': 27,'i': 28,'t': 29,'T': 30,'n': 31,'5': 32,'+': 33,'b': 34,'9': 35,
    'D': 36,'S': 37,'<sos>': 38,'<eos>': 39,'<pad>': 40}
    itos={item[1]:item[0] for item in stoi.items()}


    # Hyperparameters
    with open('configs.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfgs = yaml.load(file, Loader=yaml.FullLoader)

    vocab_size = len(vocab) + 10 # TODO: remove
    embed_size = cfgs['embed_size']
    attention_dim = cfgs['attention_dim']
    encoder_dim = cfgs['encoder_dim']
    decoder_dim = cfgs['decoder_dim']
    num_epochs = cfgs['num_epochs']
    print_every = cfgs['print_every']
    batch_size = cfgs['batch_size']
    num_workers = cfgs['num_workers']
    learning_rate = cfgs['learning_rate']
    pad_idx = stoi["<pad>"]


    # Main model
    model = EncoderDecodertrain18(
        embed_size=embed_size,
        vocab_size = vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim
    )
    model.to(device)    

    dataset_train = InputDatasetTest(train, transform, stoi)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)        
    criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])    

    for epoch in tqdm(range(num_epochs)):   
        for i, (images, inchis) in enumerate(dataloader_train):
            images, inchis = images.to(device), inchis.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward           
            outputs = model(images, inchis)

            # Calculate the batch loss.
            targets = inchis[:, 1:].to(device)
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            if (i+1) % print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))
                
                #generate the caption
                model.eval()
                with torch.no_grad():
                    dataiter = iter(dataloader_train)
                    img,_ = next(dataiter)
                    features = model.encoder(img[0:1].to(device))
                    caps = model.decoder.generate_caption(features, itos=itos, stoi=stoi)
                    captions = tensor_to_captions(caps, stoi, itos)
                    print(captions)
                    
                model.train()
            
        #save the latest model
        save_model(model,epoch,
            embed_size=200,
            vocab_size = len(vocab),
            attention_dim=300,
            encoder_dim=512,
            decoder_dim=300)