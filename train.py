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
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch_edit_distance import levenshtein_distance

from data import InputDatasetTest
from loss import LabelSmoothCrossEntropyLoss
from util import CapsCollate, get_train_file_path, get_test_file_path, tensor_to_captions, save_model, transform
from models.resnet_lstm import EncoderDecodertrain18
from models.satrn import SATRNTrnasformer


if __name__ == "__main__":
    # Get train and test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()


    # Load training dataset
    data = pd.read_csv('./input/train_labels.csv')
    print(f'train.shape: {data.shape}')


    # Make vocab
    # words=set()
    # for st in data['InChI']:
    #     words.update(set(st))
    # print(f'words length: {len(words)}')
    # words = 

    # vocab=list(words)
    # vocab.append('<sos>')
    # vocab.append('<eos>')
    # vocab.append('<pad>')
    stoi={'C': 0,')': 1,'P': 2,'l': 3,'=': 4,'3': 5,'N': 6,'I': 7,'2': 8,'6': 9,'H': 10,'4': 11,'F': 12,'0': 13,'1': 14,'-': 15,'O': 16,'8': 17,
    ',': 18,'B': 19,'(': 20,'7': 21,'r': 22,'/': 23,'m': 24,'c': 25,'s': 26,'h': 27,'i': 28,'t': 29,'T': 30,'n': 31,'5': 32,'+': 33,'b': 34,'9': 35, 'D': 36,'S': 37, 'Br': 38, 'Cl': 39, 'Si': 40, '<sos>': 41,'<eos>': 42,'<pad>': 43}
    itos={item[1]:item[0] for item in stoi.items()}


    # Hyperparameters
    with open('configs.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfgs = yaml.load(file, Loader=yaml.FullLoader)

    vocab_size = len(stoi) # TODO: remove
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
    is_sampling_mode = cfgs['is_sampling_mode']
    sample_size = cfgs['sample_size']    

    if is_sampling_mode:
        data = data.sample(n=sample_size, random_state=1)
        data = data.reset_index()
    else:
        sample_size = len(data)

    L = len(data)
    train_size = int(0.9999 * L)
    val_size = L - train_size
    print(f'Training size: {train_size}, Validation size: {val_size}')
    
    dataset = InputDatasetTest(data, transform, stoi)
    dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
   
    #dataset_train = InputDatasetTest(train, transform, stoi)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True),
        drop_last=True)

    #dataset_val = InputDatasetTest(val, transform, stoi)
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True),
        drop_last=True)        

    #Main model
    model = EncoderDecodertrain18(
        embed_size=embed_size,
        vocab_size = vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim
    )

    # model = SATRNTrnasformer(
    #     ntoken=vocab_size,
    #     encoder_dim=encoder_dim,
    #     #nhead=attention_dim,
    #     nhead=8,
    #     dim_feedforward=1024,
    #     nlayers=1,
    #     seq_length=156,
    #     decoder_dim=decoder_dim,
    #     stoi=stoi,
    #     batch_size=batch_size,
    #     embed_size=embed_size
    # )
    model.to(device)

    # # TODO
    model.load_state_dict(
        torch.load('./attention_model_state_vgg16_satrn_70500.pth')
        ['state_dict']
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)        
    criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
    #criterion = LabelSmoothCrossEntropyLoss(smoothing=0.3)
    i = 70500 # TODO

    for epoch in tqdm(range(num_epochs)):   
        for _, (images, inchis, _) in enumerate(tqdm(dataloader_train)):
            images, inchis = images.to(device), inchis.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            # Transformer
            outputs = model(images, inchis).to(device)
            targets = inchis[:, 1:].to(device)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            # For ResNet+LSTM
            # outputs = model(images, inchis).to(device)
            # targets = inchis[:, 1:].to(device)
            # loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            # For m generation
            # outputs = model(images, inchis).to(device)
            # targets = inchis.to(device)
            # loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))


            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            if (i) % print_every == 0:
                #print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))
                # generate the caption
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for _, (val_images, val_inchis, _) in enumerate(dataloader_val):
                        val_images, val_inchis = val_images.to(device), val_inchis.to(device)

                        # Transformer
                        val_outputs = model(val_images, val_inchis)
                        val_ouputs = val_outputs.to(device)
                        val_targets = val_inchis[:, 1:].to(device)

                        # ResNet + LSTM
                        # val_outputs = model(val_images, val_inchis)
                        # val_ouputs = val_outputs.to(device)
                        # val_targets = val_inchis[:, 1:].to(device)

                        # For m generation
                        # val_outputs = model(val_images, val_inchis).to(device)
                        # val_targets = val_inchis.to(device)
                        # val_loss = criterion(val_outputs.reshape(-1, vocab_size), val_targets.reshape(-1))                        
                        
                        val_loss += criterion(val_outputs.to(device).reshape(-1, vocab_size),
                                              val_targets.reshape(-1))

                    # Transformer
                    captions = model.evaluate(val_images, stoi, itos)
                    captions = tensor_to_captions(captions, stoi, itos)           

                    # ResNet + LSTM
                    # captions = model.evaluate(val_images, stoi, itos)
                    # captions = tensor_to_captions(captions, stoi, itos)
                    target_captions = tensor_to_captions(val_targets, stoi, itos)
                    
                    print(captions, target_captions)

                    val_loss /= (val_size // batch_size)
                model.train()
            
                #save the latest model
                save_model(model,epoch,
                            embed_size=embed_size,
                            vocab_size = vocab_size,
                            attention_dim=attention_dim,
                            encoder_dim=encoder_dim,
                            decoder_dim=decoder_dim )
                writer.add_scalar("loss/train", loss.item(), i)
                writer.add_scalar("loss/val", val_loss.item(), i)
                writer.flush()

            i += 1

    writer.close()
            