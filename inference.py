import csv
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
import nltk

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from data import InputDatasetTest
from loss import LabelSmoothCrossEntropyLoss
from util import CapsCollateExperiment, get_train_file_path, get_test_file_path, \
                 tensor_to_captions, save_model, transform, get_score, tensor_to_captions_multihead
from models.resnet_lstm import EncoderDecodertrain18
from models.satrn import SATRNTrnasformer
from vocab import general_to_i, t_to_i, m_to_i, s_to_i,\
                     i_to_general, i_to_t, i_to_m, i_to_s



if __name__ == "__main__":
    torch.manual_seed(42)

    # Get train and test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Load training dataset
    data = pd.read_csv('./input/sample_submission.csv', dtype=str).fillna('')
    print(f'test.shape: {data.shape}')


    # Vocabulary
    dictionaries = (general_to_i, general_to_i, m_to_i, s_to_i)
    inverse_dictionaries = (i_to_general, i_to_general, i_to_m, i_to_s)


    # Hyperparameters
    with open('configs.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        cfgs = yaml.load(file, Loader=yaml.FullLoader)

    vocab_size = len(general_to_i) # TODO: remove
    embed_size = cfgs['embed_size']
    attention_dim = cfgs['attention_dim']
    encoder_dim = cfgs['encoder_dim']
    decoder_dim = cfgs['decoder_dim']
    num_epochs = cfgs['num_epochs']
    print_every = cfgs['print_every']
    batch_size = cfgs['batch_size']
    num_workers = cfgs['num_workers']
    learning_rate = cfgs['learning_rate']
    pad_idx = general_to_i["<pad>"]
    is_sampling_mode = cfgs['is_sampling_mode']
    sample_size = cfgs['sample_size']    

    dataset = InputDatasetTest(data, transform, dictionaries, is_train=False)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)


    #Main model
    model = EncoderDecodertrain18(
        embed_size=embed_size,
        vocab_size=vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim
    )
    model.to(device)


    # Load model
    model.load_state_dict(
        torch.load('./checkpoints/attention_model_state_setting30_623600.pth')
        ['state_dict']
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
    criterion = nn.CrossEntropyLoss(ignore_index=general_to_i["<pad>"])
   
    inchis = []   
    for idx, images in enumerate(tqdm(dataloader)):
        images = images.to(device)

        # For ResNet+LSTM
        caption_tensors = model.evaluate(images, dictionaries, inverse_dictionaries)
        captions = tensor_to_captions_multihead(caption_tensors, dictionaries, inverse_dictionaries)

        # Collect InChI strings
        for note, c, h, b, t, m, s, i in zip(*list(captions)):
            inchi_str = "InChI=1S"
            # (1) Chemical note: concat as-is
            inchi_str += f"/{note}"

            # (2) C layer: insert hyphens between numbers
            inchi_str += f"/c{c}"

            # (3) H layer: concat as-is
            inchi_str += f"/h{h}"

            # (4) B layer: concat as-is if exists
            if b != '':
                inchi_str += f"/b{b}"

            # (5) T layer: concat as-is if exists
            if t != '':
                inchi_str += f"/t{t}"

            # (6) M layer: {'', '0', '0m1', '1', '1m0'} concat in integer form if exsits
            if m != '':
                inchi_str += f"/m{m}"

            # (7) S layer: {'', '1'} concat in integer form if exsits
            if s != '':
                inchi_str += f"/s{int(float(s))}"

            # (8) I layer: concat as-is if exists
            if i != '':
                inchi_str += f"/i{i}"

            # Collect
            inchis.append(inchi_str)

    output_df = pd.read_csv('./input/sample_submission.csv', dtype=str)
    output_df['InChI'] = inchis
    output_df[['image_id', 'InChI']].to_csv('./input/sample_submission_result.csv', index=False)