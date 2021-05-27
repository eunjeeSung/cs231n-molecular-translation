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
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from data import InputDatasetTest
from loss import LabelSmoothCrossEntropyLoss
from util import CapsCollateExperiment, get_train_file_path, get_test_file_path, tensor_to_captions, save_model, transform, get_score, tensor_to_captions_multihead
from models.resnet_lstm import EncoderDecodertrain18
from models.satrn import SATRNTrnasformer


if __name__ == "__main__":
    torch.manual_seed(42)

    # Get train and test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()


    # Load training dataset
    data = pd.read_csv('./input/train_labels_layered.csv', dtype=str).fillna('')
    print(f'train.shape: {data.shape}')


    # Vocabulary
    general_to_i = {'(': 0, ')': 1, '+': 2, ',': 3, '-': 4, '/b': 5, '/c': 6, '/h': 7, '/i': 8, '/m': 9, '/s': 10, '/t': 11, '0': 12, '1': 13, '10': 14, '100': 15, '101': 16, '102': 17, '103': 18, '104': 19, '105': 20, '106': 21, '107': 22, '108': 23, '109': 24, '11': 25, '110': 26, '111': 27, '112': 28, '113': 29, '114': 30, '115': 31, '116': 32, '117': 33, '118': 34, '119': 35, '12': 36, '120': 37, '121': 38, '122': 39, '123': 40, '124': 41, '125': 42, '126': 43, '127': 44, '128': 45, '129': 46, '13': 47, '130': 48, '131': 49, '132': 50, '133': 51, '134': 52, '135': 53, '136': 54, '137': 55, '138': 56, '139': 57, '14': 58, '140': 59, '141': 60, '142': 61, '143': 62, '144': 63, '145': 64, '146': 65, '147': 66, '148': 67, '149': 68, '15': 69, '150': 70, '151': 71, '152': 72, '153': 73, '154': 74, '155': 75, '156': 76, '157': 77, '158': 78, '159': 79, '16': 80, '161': 81, '163': 82, '165': 83, '167': 84, '17': 85, '18': 86, '19': 87, '2': 88, '20': 89, '21': 90, '22': 91, '23': 92, '24': 93, '25': 94, '26': 95, '27': 96, '28': 97, '29': 98, '3': 99, '30': 100, '31': 101, '32': 102, '33': 103, '34': 104, '35': 105, '36': 106, '37': 107, '38': 108, '39': 109, '4': 110, '40': 111, '41': 112, '42': 113, '43': 114, '44': 115, '45': 116, '46': 117, '47': 118, '48': 119, '49': 120, '5': 121, '50': 122, '51': 123, '52': 124, '53': 125, '54': 126, '55': 127, '56': 128, '57': 129, '58': 130, '59': 131, '6': 132, '60': 133, '61': 134, '62': 135, '63': 136, '64': 137, '65': 138, '66': 139, '67': 140, '68': 141, '69': 142, '7': 143, '70': 144, '71': 145, '72': 146, '73': 147, '74': 148, '75': 149, '76': 150, '77': 151, '78': 152, '79': 153, '8': 154, '80': 155, '81': 156, '82': 157, '83': 158, '84': 159, '85': 160, '86': 161, '87': 162, '88': 163, '89': 164, '9': 165, '90': 166, '91': 167, '92': 168, '93': 169, '94': 170, '95': 171, '96': 172, '97': 173, '98': 174, '99': 175, 'B': 176, 'Br': 177, 'C': 178, 'Cl': 179, 'D': 180, 'F': 181, 'H': 182, 'I': 183, 'N': 184, 'O': 185, 'P': 186, 'S': 187, 'Si': 188, 'T': 189, 't': 190, 'h': 191, '<sos>': 192, '<eos>': 193, '<pad>': 194}
    i_to_general = {item[1]:item[0] for item in general_to_i.items()}

    t_to_i = {'39-': 0, '40-': 1, '12-t10-': 2, '8+t3-': 3, '16+t12-': 4, '30+': 5, '14+t8-': 6, '53+': 7, '11-t8-': 8, '26-': 9, '15+': 10, '4+': 11, '43+': 12, '12-t6-': 13, '15-t9-': 14, '18-t13-': 15, '16+': 16, '8-t7-': 17, '27-': 18, '32-': 19, '22+': 20, '38-': 21, '13-t9-': 22, '11-t5-': 23, '51-': 24, '60-': 25, '36-': 26, '4-': 27, '31+': 28, '21-t14-': 29, '17+': 30, '28-': 31, '6-': 32, '43-': 33, '42+': 34, '9+': 35, '8-t5-': 36, '56+': 37, '40+': 38, '23+': 39, '6+t1-': 40, '36+': 41, '11-t6-': 42, '38+': 43, '62-': 44, '14-t13-': 45, '63-': 46, '47-': 47, '22-t9-': 48, '5+t2-': 49, '12+': 50, '8-t2-': 51, '48-': 52, '25-t15-': 53, '41-': 54, '13+t10-': 55, '57-': 56, '50-': 57, '12-': 58, '21-': 59, '10-t9-': 60, '17-': 61, '9-t7-': 62, '14-t11-': 63, '8-': 64, '2+': 65, '55-': 66, '19-t6-': 67, '59-': 68, '2-': 69, '19+t12-': 70, '3+': 71, '5-': 72, '37+': 73, '66-': 74, '17-t12-': 75, '30-': 76, '9-': 77, '3-': 78, '26+': 79, '6+': 80, '11-t9-': 81, '19-': 82, '44-': 83, '34+': 84, '61-': 85, '48+': 86, '10+': 87, '47+': 88, '10+t8-': 89, '56-': 90, '19+': 91, '34-': 92, '28+': 93, '35-': 94, '7-': 95, '25+': 96, '13-t5-': 97, '6-t5-': 98, '41+': 99, '20+': 100, '65-': 101, '31-': 102, '50+': 103, '10+t4-': 104, '68-': 105, '54+': 106, '7+': 107, '23-t15-': 108, '4-t1-': 109, '6-t3-': 110, '6+t3-': 111, '7-t4-': 112, '14+': 113, '23-': 114, '11+': 115, '46+': 116, '42-': 117, '5-t3-': 118, '64+': 119, '52-': 120, '29+': 121, '49+': 122, '37-': 123, '64-': 124, '1-': 125, '13+': 126, '7-t2-': 127, '33+': 128, '52+': 129, '5-t4-': 130, '10-t8-': 131, '29-': 132, '13-': 133, '8+': 134, '20-': 135, '32+': 136, '7-t5-': 137, '51+': 138, '15-': 139, '35+': 140, '39+': 141, '10-': 142, '27+': 143, '46-': 144, '5+': 145, '16-t9-': 146, '17-t13-': 147, '16-': 148, '33-': 149, '45-': 150, '54-': 151, '18-': 152, '24+': 153, '24-': 154, '21+': 155, '25-': 156, '58-': 157, '18+': 158, '44+': 159, '28+t17-': 160, '22-': 161, '17-t10-': 162, '45+': 163, '14-': 164, '49-': 165, '53-': 166, '11-': 167, '<sos>': 191, '<eos>': 192, '<pad>': 193}
    i_to_t = {item[1]:item[0] for item in t_to_i.items()}

    m_to_i = {'': 0, '0': 1, '0m1': 2, '1': 3, '1m0': 4}
    i_to_m = {item[1]:item[0] for item in m_to_i.items()}

    s_to_i = {'': 0, '1.0': 1}
    i_to_s = {item[1]:item[0] for item in s_to_i.items()}

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

    if is_sampling_mode:
        data = data.sample(n=sample_size, random_state=1)
        data = data.reset_index()
    else:
        sample_size = len(data)

    L = len(data)
    train_size = int(0.9999 * L)
    val_size = L - train_size
    print(f'Training size: {train_size}, Validation size: {val_size}')
    
    dataset = InputDatasetTest(data, transform, dictionaries)
    dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
   
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=CapsCollateExperiment(pad_idx=pad_idx, batch_first=True),
        drop_last=True)

    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=CapsCollateExperiment(pad_idx=pad_idx, batch_first=True),
        drop_last=True)        

    #Main model
    model = EncoderDecodertrain18(
        embed_size=embed_size,
        vocab_size=vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim
    )
    model.to(device)

    # TODO
    model.load_state_dict(
        torch.load('./checkpoints/attention_model_state_aug_340800.pth')
        ['state_dict']
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
    criterion = nn.CrossEntropyLoss(ignore_index=general_to_i["<pad>"])
    i = 340800 # TODO

    for epoch in range(num_epochs):   
        for _, (images, vecs) in enumerate(tqdm(dataloader_train)):
            images = images.to(device)
            vecs = [v.to(device) for v in vecs]

            # Zero the gradients.
            optimizer.zero_grad()

            # For ResNet+LSTM
            note_outs, c_outs, h_outs, b_outs, t_outs, m_outs, s_outs, i_outs = model(images, vecs)
            (note_vecs, c_vecs, h_vecs, b_vecs, t_vecs, m_vecs, s_vecs, i_vecs) = vecs
            note_vecs, c_vecs, h_vecs, b_vecs, t_vecs, i_vecs = \
                note_vecs[:, 1:], c_vecs[:, 1:], h_vecs[:, 1:], b_vecs[:, 1:], t_vecs[:, 1:], i_vecs[:, 1:]            

            note_loss = criterion(note_outs.reshape(-1, vocab_size), note_vecs.reshape(-1))
            c_loss = criterion(c_outs.reshape(-1, vocab_size), c_vecs.reshape(-1))
            h_loss = criterion(h_outs.reshape(-1, vocab_size), h_vecs.reshape(-1))
            b_loss = criterion(b_outs.reshape(-1, vocab_size), b_vecs.reshape(-1))
            t_loss = criterion(t_outs.reshape(-1, vocab_size), t_vecs.reshape(-1))
            m_loss = criterion(m_outs, m_vecs.reshape(-1))
            s_loss = criterion(s_outs, s_vecs.reshape(-1))
            i_loss = criterion(i_outs.reshape(-1, vocab_size), i_vecs.reshape(-1))

            loss = note_loss + c_loss + h_loss + b_loss + t_loss + m_loss + s_loss + i_loss

            # Backward pass.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Update the parameters in the optimizer.
            optimizer.step()

            if (i) % print_every == 0:
                # generate the caption
                model.eval()
                sample_images, sample_inchis = None, None
                with torch.no_grad():
                    val_loss, val_num = 0, 0
                    for _, (val_images, val_vecs) in enumerate(dataloader_val):
                        val_images = val_images.to(device)
                        val_vecs = [v.to(device) for v in val_vecs]

                        # ResNet + LSTM
                        val_outputs = model(val_images, val_vecs)
                        (val_note_outs, val_c_outs, val_h_outs, val_b_outs, val_t_outs, val_m_outs, val_s_outs, val_i_outs) = val_outputs
                        (val_note_vecs, val_c_vecs, val_h_vecs, val_b_vecs, val_t_vecs, val_m_vecs, val_s_vecs, val_i_vecs) = val_vecs
                        val_note_vecs, val_c_vecs, val_h_vecs, val_b_vecs, val_t_vecs, val_i_vecs = \
                            val_note_vecs[:, 1:], val_c_vecs[:, 1:], val_h_vecs[:, 1:], val_b_vecs[:, 1:], val_t_vecs[:, 1:], val_i_vecs[:, 1:]
                        val_vecs = (val_note_vecs, val_c_vecs, val_h_vecs, val_b_vecs, val_t_vecs, val_m_vecs, val_s_vecs, val_i_vecs)

                        val_loss += criterion(val_note_outs.reshape(-1, vocab_size), val_note_vecs.reshape(-1))
                        val_loss += criterion(val_c_outs.reshape(-1, vocab_size), val_c_vecs.reshape(-1))
                        val_loss += criterion(val_h_outs.reshape(-1, vocab_size), val_h_vecs.reshape(-1))
                        val_loss += criterion(val_b_outs.reshape(-1, vocab_size), val_b_vecs.reshape(-1))
                        val_loss += criterion(val_t_outs.reshape(-1, vocab_size), val_t_vecs.reshape(-1))
                        val_loss += criterion(val_m_outs, val_m_vecs.reshape(-1))
                        val_loss += criterion(val_s_outs, val_s_vecs.reshape(-1))
                        val_loss += criterion(val_i_outs.reshape(-1, vocab_size), val_i_vecs.reshape(-1))
                        
                        val_num += 1

                        sample_images, sample_targets = val_images, val_vecs

                    # ResNet + LSTM
                    caption_tensors = model.evaluate(sample_images[:4], dictionaries, inverse_dictionaries)
                    captions = tensor_to_captions_multihead(caption_tensors, dictionaries, inverse_dictionaries)
                    target_captions = tensor_to_captions_multihead(sample_targets, dictionaries, inverse_dictionaries)
                    lev_dist = get_score([captions[1][0]], [target_captions[1][0]])
                    
                    for out_layer, target_layer in zip(captions, target_captions):
                        print(f"Pred: [{out_layer[0]}] / Target: [{target_layer[0]}]")

                    val_loss /= val_num
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
                writer.add_scalar("distance/val", lev_dist, i)
                writer.flush()

            i += 1

    writer.close()
            