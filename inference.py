import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import pandas as pd
import yaml
from tqdm import tqdm

from data import InputDatasetTest
from models.resnet_lstm import EncoderDecodertrain18
from util import CapsCollate, tensor_to_captions, transform


if __name__ == "__main__":
    with open('configs.yaml') as file:
        cfgs = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = pd.read_csv('./input/sample_submission.csv')            
    train = pd.read_csv('./input/train_labels.csv')

    # Make vocab
    # TODO: store vocabs
    words=set()
    for st in train['InChI']:
        words.update(set(st))
    print(f'words length: {len(words)}')
    del(train)

    vocab=list(words)
    vocab.append('<sos>')
    vocab.append('<eos>')
    vocab.append('<pad>')
    stoi={'C': 0,')': 1,'P': 2,'l': 3,'=': 4,'3': 5,'N': 6,'I': 7,'2': 8,'6': 9,'H': 10,'4': 11,'F': 12,'0': 13,'1': 14,'-': 15,'O': 16,'8': 17,
    ',': 18,'B': 19,'(': 20,'7': 21,'r': 22,'/': 23,'m': 24,'c': 25,'s': 26,'h': 27,'i': 28,'t': 29,'T': 30,'n': 31,'5': 32,'+': 33,'b': 34,'9': 35,
    'D': 36,'S': 37,'<sos>': 38,'<eos>': 39,'<pad>': 40}
    itos={item[1]:item[0] for item in stoi.items()}

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
    # TODO: implement official loss function
    criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
    is_sampling_mode = cfgs['is_sampling_mode']
    sample_size = cfgs['sample_size']

    if is_sampling_mode:
        test = test.sample(n=sample_size, random_state=1)
        test = test.reset_index()


    # Model
    MODEL_PATH = './checkpoints/sample.pth'    
    model = EncoderDecodertrain18(
        embed_size=embed_size,
        vocab_size = vocab_size,
        attention_dim=attention_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim    
    )
    model.to(device)
    # TODO: load hyperparameters
    model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    
    dataset_test = InputDatasetTest(test, transform, stoi, is_train=False)
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True))

    model.eval()
    losses = []
    with torch.no_grad():
        for i, (images, inchis, indices) in enumerate(tqdm(dataloader_test)):
            images, inchis = images.to(device), inchis.to(device)
            features = model.encoder(images)
            caps = model.decoder.generate_caption(features, stoi=stoi, itos=itos)
            captions = tensor_to_captions(caps, stoi=stoi, itos=itos)
            test['InChI'].loc[indices] = captions
            if i % print_every == 0:
                # TODO: only pass caption length to the model
                outputs = model(images, inchis)
                targets = inchis[:, 1:].to(device)
                loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
                losses.append(loss)
                print(f'Example output: {captions[0]} / Example loss: {loss}')

    print(f'Average loss: {sum(losses) / len(losses)}, #loss: {len(losses)}')
    output_cols = ['image_id', 'InChI']
    test[output_cols].to_csv('submission.csv',index=False)
    print(test[output_cols].head())